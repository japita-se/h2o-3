package hex.svm;

import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.Log;

public class PrimalDualIPM {
  
  static void solve(Frame rbicf, Vec label, Params params) {
    checkLabel(label);

    final double c_pos = params._weight_positive * params._hyper_parm;
    final double c_neg = params._weight_negative * params._hyper_parm;
    final long num_constraints = rbicf.numRows() * 2; 

    Vec x = label.makeZero();
    Frame initFrame = new InitTask(c_pos, c_neg).doAll(new byte[]{Vec.T_NUM, Vec.T_NUM}, label).outputFrame();
    Vec la = initFrame.vec(0);
    Vec xi = initFrame.vec(1);

    double nu = 0;
    
    for (int iter = 0; iter < params._max_iter; iter++) {
      double eta = computeSurrogateGap(c_pos, c_neg, label, x, la, xi);
      double t = (params._mu_factor * num_constraints) / eta;
      Log.debug("sgap: " + eta + " t: " + t);

      Vec z = computePartialZ(rbicf, x, params._tradeoff);
      CheckConvergenceTask cct = new CheckConvergenceTask(nu).doAll(z, label, la, x, xi);
      if (cct._resp <= params._feasible_threshold && cct._resd <= params._feasible_threshold 
              && eta <= params._sgap_bound) {
        break;
      }
      
      Frame working = new UpdateVarsTask(c_pos, c_neg, params._x_epsilon, t)
              .doAll(new byte[]{Vec.T_NUM, Vec.T_NUM, Vec.T_NUM, Vec.T_NUM, Vec.T_NUM}, z, label, la, x, xi)
              .outputFrame(new String[]{"tlx", "tux", "xilx", "laux", "d"}, null); 

      LLMatrix icfA = MatrixUtils.productMM(rbicf, working.vec("d"));
      LLMatrix lra = MatrixUtils.cf(icfA);

      final double dnu = computeDeltaNu(rbicf, working.vec("d"), label, z, x, lra);
      Vec dx = computeDeltaX(rbicf, working.vec("d"), label, dnu, lra, z);

      LineSearchTask lst = new LineSearchTask(c_pos, c_neg).doAll(new byte[]{Vec.T_NUM, Vec.T_NUM}, label, working.vec("tlx"), working.vec("tux"), working.vec("xilx"), working.vec("laux"), xi, la, dx, x);
      Frame lstFrame = lst.outputFrame(new String[]{"dxi", "dla"}, null);

      new MakeStepTask(lst._ap, lst._ad).doAll(x, dx, xi, lstFrame.vec("dxi"), la, lstFrame.vec("dla"));

      nu += lst._ad * dnu;
    }
  }

  static class MakeStepTask extends MRTask<MakeStepTask> {
    double _ap;
    double _ad;

    MakeStepTask(double ap, double ad) {
      _ap = ap;
      _ad = ad;
    }
    
    @Override
    public void map(Chunk[] cs) {
      map(cs[0], cs[1], cs[2], cs[3], cs[4], cs[5]);
    }

    public void map(Chunk x, Chunk dx, Chunk xi, Chunk dxi, Chunk la, Chunk dla) {
      for (int i = 0; i < x._len; i++) {
        x.set(i, x.atd(i) + (_ap * dx.atd(i)));
        xi.set(i, xi.atd(i) + (_ad * dxi.atd(i)));
        la.set(i, la.atd(i) + (_ad * dla.atd(i)));
      }
    }
    
  }

  static class LineSearchTask extends MRTask<LineSearchTask> {
    private final double _c_pos;
    private final double _c_neg;

    private double _ap;
    private double _ad;
    
    LineSearchTask(double c_pos, double c_neg) {
      _c_pos = c_pos;
      _c_neg = c_neg;
    }

    public void map(Chunk[] cs, NewChunk[] ncs) {
      map(cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], cs[7], cs[8], ncs[0], ncs[1]);
    }

    private void map(Chunk label, Chunk tlx, Chunk tux, Chunk xilx, Chunk laux, Chunk xi, Chunk la, Chunk dx, Chunk x, NewChunk dxiC, NewChunk dlaC) {
      double[] dxi = new double[tlx._len];
      double[] dla = new double[tlx._len];
      for (int i = 0; i < dxi.length; ++i) {
        dxi[i] = tlx.atd(i) - xilx.atd(i) * dx.atd(i) - xi.atd(i);
        dxiC.addNum(dxi[i]);
        dla[i] = tux.atd(i) + laux.atd(i) * dx.atd(i) - la.atd(i);
        dlaC.addNum(dla[i]);
      }
      double ap = Double.MAX_VALUE;
      double ad = Double.MAX_VALUE;
      for (int i = 0; i < dxi.length; i++) {
        double c = (label.atd(i) > 0.0) ? _c_pos : _c_neg;
        if (dx.atd(i) > 0.0) {
          ap = Math.min(ap, (c - x.atd(i)) / dx.atd(i));
        }
        if (dx.atd(i) < 0.0) {
          ap = Math.min(ap, -x.atd(i)/dx.atd(i));
        }
        if (dxi[i] < 0.0) {
          ad = Math.min(ad, -xi.atd(i) / dxi[i]);
        }
        if (dla[i] < 0.0) {
          ad = Math.min(ad, -la.atd(i) / dla[i]);
        }
      }
      _ap = ap;
      _ad = ad;
    }

    @Override
    public void reduce(LineSearchTask mrt) {
      _ap = Math.min(_ap, mrt._ap);
      _ad = Math.min(_ad, mrt._ad);
    }

    @Override
    public void postGlobal() {
      _ap = Math.min(_ap, 1.0) * 0.99;
      _ad = Math.min(_ad, 1.0) * 0.99;
    }
  }
  
  private static void checkLabel(Vec label) {
    if (label.min() != -1 || label.max() != 1)
      throw new IllegalArgumentException("Expected a binary response encoded as +1/-1");
  }
  
  static class UpdateVarsTask extends MRTask<UpdateVarsTask> {
    private final double _c_pos;
    private final double _c_neg;
    private final double _epsilon_x;
    private final double _t;

    UpdateVarsTask(double c_pos, double c_neg, double epsilon_x, double t) {
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
      map(cs[0], cs[1], cs[2], cs[3], cs[4]);
    }

    public void map(Chunk z, Chunk label, Chunk la, Chunk x, Chunk xi) {
      for (int i = 0; i < z._len; i++) {
        double zi = z.atd(i);
        zi += _nu * (label.atd(i) > 0 ? 1 : -1) - 1.0;
        double temp = la.atd(i) - xi.atd(i) + zi;
        z.set(i, zi);
        _resd += temp * temp;
        _resp += label.atd(i) * x.atd(i);
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

  private static Vec computePartialZ(Frame rbicf, Vec x, final double tradeoff) {
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
  }

  private static double computeSurrogateGap(double c_pos, double c_neg, Vec response, Vec x, Vec la, Vec xi) {
    return new SurrogateGapTask(c_pos, c_neg).doAll(new Vec[]{response, x, la, xi})._sum;
  }

  static class SurrogateGapTask extends MRTask<SurrogateGapTask> {
    private final double _c_pos;
    private final double _c_neg;
    
    // OUT
    private double _sum;

    SurrogateGapTask(double c_pos, double c_neg) {
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

    InitTask(double c_pos, double c_neg) {
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

  private static Vec computeDeltaX(Frame icf, Vec d, Vec label, final double dnu, LLMatrix lra, Vec z) {
    Vec tz = new MRTask() {
      @Override
      public void map(Chunk z, Chunk label, NewChunk tz) {
        for (int i = 0; i < z._len; i++) {
          tz.addNum(z.atd(i) - dnu * label.atd(i));
        }
      }
    }.doAll(Vec.T_NUM, z, label).outputFrame().anyVec();
    return linearSolveViaICFCol(icf, d, tz, lra);
  }
  
  private static double computeDeltaNu(Frame icf, Vec d, Vec label, Vec z, Vec x, LLMatrix lra) {
    Vec tw = linearSolveViaICFCol(icf, d, z, lra);
    Vec tl = linearSolveViaICFCol(icf, d, label, lra);
    DeltaNuTask dnt = new DeltaNuTask().doAll(label, tw, tl, x);
    return dnt._sum1 / dnt._sum2;
  }
  
  static class DeltaNuTask extends MRTask<DeltaNuTask> {
    double _sum1;
    double _sum2;

    @Override
    public void map(Chunk[] cs) {
      map(cs[0], cs[1], cs[2], cs[3]);
    }

    public void map(Chunk label, Chunk tw, Chunk tl, Chunk x) {
      for (int i = 0; i < label._len; i++) {
        _sum1 += label.atd(i) * (tw.atd(i) + x.atd(i));
        _sum2 += label.atd(i) * tl.atd(i);
      }
    }

    @Override
    public void reduce(DeltaNuTask mrt) {
      _sum1 += mrt._sum1;
      _sum2 += mrt._sum2;
    }
  }
  
  private static Vec linearSolveViaICFCol(Frame icf, Vec d, Vec b, LLMatrix lra) {
    LSHelper1 lsh = new LSHelper1().doAll(Vec.T_NUM, ArrayUtils.append(icf.vecs(), d, b));
    Vec x = lsh.outputFrame().anyVec();
    final double[] vz = lsh._row;
    double[] ty = new double[vz.length];
    MatrixUtils.cholForwardSub(lra, vz, ty);
    MatrixUtils.cholBackwardSub(lra, ty, vz);
    new MRTask() {
      @Override
      public void map(Chunk[] cs) {
        final int p = cs.length - 2;
        Chunk d = cs[p];
        Chunk x = cs[p + 1];
        for (int i = 0; i < cs[0]._len; i++) {
          double s = 0.0;
          for (int j = 0; j < p; j++) {
            s += cs[j].atd(i) * vz[j] * d.atd(i);
          }
          x.set(i, x.atd(i) - s);
        }
      }
    }.doAll(ArrayUtils.append(icf.vecs(), d, x));
    return x;
  }

  static class LSHelper1 extends MRTask<LSHelper1> {
    double[] _row;
    @Override
    public void map(Chunk[] cs, NewChunk nc) {
      final int p = cs.length - 2;
      _row = new double[p];
      Chunk d = cs[p];
      Chunk b = cs[p + 1];
      double[] z = new double[cs[0]._len];
      for (int i = 0; i < z.length; i++) {
        z[i] = b.atd(i) * d.atd(i);
      }
      for (int j = 0; j < p; j++) {
        double s = 0.0;
        for (int i = 0; i < z.length; i++) {
          s += cs[j].atd(i) * z[i];
        }
        _row[j] = s;
      }
      for (double zi : z) {
        nc.addNum(zi);
      }
    }

    @Override
    public void reduce(LSHelper1 mrt) {
      ArrayUtils.add(_row, mrt._row);
    }
  }

  static class Params {
    int _max_iter = 30;
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
