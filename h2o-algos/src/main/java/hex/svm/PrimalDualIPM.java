package hex.svm;

import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.Log;

public class PrimalDualIPM {
  
  static void solve(Frame rbicf, Vec response, Params params) {
    final double c_pos = params._weight_positive * params._hyper_parm;
    final double c_neg = params._weight_negative * params._hyper_parm;
    final long num_constraints = rbicf.numRows() * 2; 

    Vec x = response.makeZero();
    Frame initFrame = new InitTask(c_pos, c_neg).doAll(new byte[]{Vec.T_NUM, Vec.T_NUM}, response).outputFrame();
    Vec la = initFrame.vec(0);
    Vec xi = initFrame.vec(1);

    double nu = 0;
    
    for (int iter = 0; iter < params._max_iter; iter++) {
      double eta = computeSurrogateGap(c_pos, c_neg, response, x, la, xi);
      double t = (params._mu_factor * num_constraints) / eta;
      Log.debug("sgap: " + eta + " t: " + t);

      Vec z = computePartialZ(rbicf, x, params._tradeoff);
      CheckConvergenceTask cct = new CheckConvergenceTask(nu).doAll(z, response, la, xi);
      if (cct._resp < params._feasible_threshold && cct._resd < params._feasible_threshold 
              && eta < params._sgap_bound) {
        break;
      }
      
      Frame working = new UpdateVarsTask(c_pos, c_neg, params._x_epsilon, t)
              .doAll(new byte[]{Vec.T_NUM, Vec.T_NUM, Vec.T_NUM, Vec.T_NUM, Vec.T_NUM}, z, response, la, x, xi)
              .outputFrame(new String[]{"tlx", "tux", "xilx", "laux", "d"}, null); 

      LLMatrix icfA = MatrixUtils.productMM(rbicf, working.vec("d"));
      LLMatrix lra = MatrixUtils.cf(icfA);

      System.out.println(lra);
      // double dnu = computeDeltaNu(rbicf, working.vec("d"), response, z, x, lra);
    }
  }

  static class UpdateVarsTask extends MRTask<UpdateVarsTask> {
    private final double _c_pos;
    private final double _c_neg;
    private final double _epsilon_x;
    private final double _t;

    // OUT
    private double _sum;

    public UpdateVarsTask(double c_pos, double c_neg, double epsilon_x, double t) {
      _c_pos = c_pos;
      _c_neg = c_neg;
      _epsilon_x = epsilon_x;
      _t = t;
    }

    @Override
    public void map(Chunk[] cs, NewChunk[] ncs) {
      map(cs[0], cs[1], cs[2], cs[3], cs[4], ncs[0], ncs[1], ncs[2], ncs[3], ncs[4]);
    }
    
    private void map(Chunk z, Chunk label, Chunk la, Chunk x, Chunk xi,
                     NewChunk tlx, NewChunk tux, NewChunk xilx, NewChunk laux, NewChunk d) {
      for (int i = 0; i < z._len; i++) {
        double c = (label.atd(i) > 0) ?_c_pos : _c_neg;
        double m_lx = Math.max(x.atd(i), _epsilon_x);
        double m_ux = Math.max(c - x.atd(i), _epsilon_x);
        double tlxi = 1.0 / (_t * m_lx);
        double tuxi = 1.0 / (_t * m_ux);
        tlx.addNum(tlxi);
        tux.addNum(tuxi);
        
        double xilxi = Math.max(xi.atd(i) / m_lx, _epsilon_x);
        double lauxi = Math.max(la.atd(i) / m_ux, _epsilon_x);
        d.addNum(1.0 / (xilxi + lauxi));
        xilx.addNum(xilxi);
        laux.addNum(lauxi);
        
        z.set(i, tlxi - tuxi - z.atd(i));
      }
    }
  }
  
  static class CheckConvergenceTask extends MRTask<CheckConvergenceTask> {
    private final double _nu;
    // OUT
    double _resd;
    double _resp;

    CheckConvergenceTask(double nu) {
      _nu = nu;
    }

    @Override
    public void map(Chunk[] cs) {
      map(cs[0], cs[1], cs[2], cs[3]);
    }

    public void map(Chunk z, Chunk label, Chunk la, Chunk xi) {
      for (int i = 0; i < z._len; i++) {
        double zi = z.atd(i);
        zi += _nu * (label.atd(i) > 0 ? 1 : -1) - 1.0;
        double temp = la.atd(i) - xi.atd(i) + zi;
        z.set(i, zi);
        _resd += temp * temp;
        _resp += label.atd(i) * xi.atd(i);
      }
    }

    @Override
    public void reduce(CheckConvergenceTask mrt) {
      _resd += mrt._resd;
      _resp += mrt._resp;
    }

    @Override
    protected void postGlobal() {
      _resp = Math.abs(_resp);
      _resd = Math.sqrt(_resd);
    }
  }

  static Vec computePartialZ(Frame rbicf, Vec x, final double tradeoff) {
    final Vec[] vecs = ArrayUtils.append(rbicf.vecs(), x);
    final double vz[] = new MatrixMultVecTask().doAll(vecs)._row;
    return new MRTask() {
      @Override
      public void map(Chunk[] cs, NewChunk z) {
        final int p = cs.length - 1;
        final Chunk x = cs[p];
        for (int i = 0; i < cs[0]._len; i++) {
          double s = 0;
          for (int j = 0; j < p; j++) {
            s += cs[j].atd(i) * vz[j];
          }
          z.addNum(s - tradeoff * x.atd(i));
        }
      }
    }.doAll(Vec.T_NUM, vecs).outputFrame().anyVec();
  }
  
  static class MatrixMultVecTask extends MRTask<MatrixMultVecTask> {
    double[] _row;

    @Override
    public void map(Chunk[] cs) {
      final int p = cs.length - 1;
      final Chunk x = cs[p];
      _row = new double[p];
      for (int j = 0; j < p; ++j) {
        double sum = 0.0;
        for (int i = 0; i < cs[0]._len; i++) {
          sum += cs[j].atd(i) * x.atd(i);
        }
        _row[j] = sum;
      }
    }

    @Override
    public void reduce(MatrixMultVecTask mrt) {
      ArrayUtils.add(_row, mrt._row);
    }

    /*
        register int i, j;
  int p = icf.GetNumCols();
  double *vz = new double[p];
  double *vzpart = new double[p];
  // form vz = V^T*x
  memset(vzpart, 0, sizeof(vzpart[0]) * p);
  double sum;
  for (j = 0; j < p; ++j) {
    sum = 0.0;
    for (i = 0; i < local_num_rows; ++i) {
      sum += icf.Get(i, j) * x[i];
    }
    vzpart[j] = sum;
  }
  ParallelInterface *mpi = ParallelInterface::GetParallelInterface();
  mpi->AllReduce(vzpart, vz, p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // form z = V*vz
  for (i = 0; i < local_num_rows; ++i) {
    // Get a piece of inner product
    sum = 0.0;
    for (j = 0; j < p; ++j) {
      sum += icf.Get(i, j) * vz[j];
    }
    z[i] = sum - to * x[i];
  }

  delete [] vz;
  delete [] vzpart;
  return 0;
       */
  } 
  
  private static double computeSurrogateGap(double c_pos, double c_neg, Vec response, Vec x, Vec la, Vec xi) {
    return new SurrogateGapTask(c_pos, c_neg).doAll(new Vec[]{response, x, la, xi})._sum;
  }

  static class SurrogateGapTask extends MRTask<SurrogateGapTask> {
    private final double _c_pos;
    private final double _c_neg;
    
    // OUT
    private double _sum;

    public SurrogateGapTask(double c_pos, double c_neg) {
      _c_pos = c_pos;
      _c_neg = c_neg;
    }

    @Override
    public void map(Chunk[] cs) {
      _sum = map(cs[0], cs[1], cs[2], cs[3]);
    }

    public double map(Chunk label, Chunk x, Chunk la, Chunk xi) {
      double s = 0;
      for (int i = 0; i < x._len; i++) {
        double c = (label.atd(i) > 0.0) ? _c_pos : _c_neg;
        s += la.atd(i) * c;
      }
      for (int i = 0; i < x._len; i++) {
        s += x.atd(i) * (xi.atd(i) - la.atd(i));
      }
      return s;
    }

    @Override
    public void reduce(SurrogateGapTask mrt) {
      _sum += mrt._sum;
    }
  }
  
  static class InitTask extends MRTask<InitTask> {
    private final double _c_pos;
    private final double _c_neg;

    public InitTask(double c_pos, double c_neg) {
      _c_pos = c_pos;
      _c_neg = c_neg;
    }

    @Override
    public void map(Chunk[] cs, NewChunk[] nc) {
      Chunk label = cs[0];
      for (int i = 0; i < label._len; i++) {
        double c = ((label.atd(i) > 0) ? _c_pos : _c_neg) / 10;
        nc[0].addNum(c);
        nc[1].addNum(c);
      }
    }
  }
  
  static class Params {
    int _max_iter = 1;
    double _weight_positive = 1.0;
    double _weight_negative = 1.0;
    double _hyper_parm = 1.0;
    double _mu_factor = 10.0;
    double _tradeoff = 0;
    double _feasible_threshold = 1.0e-3;
    double _sgap_bound = 1.0e-3;
    double _x_epsilon = 1.0e-9;
  }
  
    
}
