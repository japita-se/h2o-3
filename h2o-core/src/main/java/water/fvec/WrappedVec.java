package water.fvec;

import water.*;

/**
 * A simple wrapper over another Vec.  Transforms either data values or rows.
 */
abstract class WrappedVec extends Vec {
  /** A key for underlying vector which contains values which are transformed by this vector. */
  final Key<Vec> _masterVecKey;
  final boolean _cascadeDelete;
  /** Cached instances of underlying vector. */
  transient Vec _masterVec;


  public WrappedVec(Key<Vec> key, int rowLayout, Key<Vec> masterVecKey ) {
    this(key, rowLayout, null, masterVecKey);
  }
  public WrappedVec(Key<Vec> key, int rowLayout, String[] domain, Key<Vec> masterVecKey) {
    this(key, rowLayout, domain, masterVecKey, false);
  }
  public WrappedVec(Key<Vec> key, int rowLayout, String[] domain, Key<Vec> masterVecKey, boolean cascadeDelete) {
    super(key, rowLayout, domain);
    _masterVecKey = masterVecKey;
    _cascadeDelete = cascadeDelete;
  }

  public Vec masterVec() {
    return _masterVec != null ? _masterVec : (_masterVec = _masterVecKey.get());
  }

  /** Map from chunk-index to Chunk.  These wrappers are making custom Chunks */
  public abstract Chunk chunkForChunkIdx(int cidx);

  @Override
  public Futures remove_impl(Futures fs) {
    super.remove_impl(fs);
    if (_cascadeDelete && _masterVecKey != null) {
      Vec masterVec = _masterVecKey.get();
      if (masterVec != null) {
        masterVec.remove(fs);
      }
    }
    return fs;
  }
}
