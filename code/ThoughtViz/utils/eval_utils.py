import os.path
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import math
import sys
from keras.utils import to_categorical

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None
_inception_sess = None


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  _init_inception()
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 1
  preds = []
  n_batches = int(math.ceil(float(len(inps)) / float(bs)))
  batch_range = range(n_batches)
  if tqdm is not None:
    batch_range = tqdm(
        batch_range,
        desc="Inception IS",
        unit="img",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
    )
  for i in batch_range:
      inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
      inp = np.concatenate(inp, 0)
      pred = _inception_sess.run(softmax, {'ExpandDims:0': inp})
      preds.append(pred)
  preds = np.concatenate(preds, 0)
  scores = []
  for i in range(splits):
    part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    scores.append(np.exp(kl))
  return np.mean(scores), np.std(scores)


# This function is called automatically.
def _init_inception():
  global softmax, _inception_sess
  if _inception_sess is not None:
    return
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  # TF2: tf.gfile removed; use local open(). Use a dedicated Graph + compat.v1.Session
  # so we do not call disable_eager_execution() (would break Keras training).
  pb = os.path.join(MODEL_DIR, 'classify_image_graph_def.pb')
  g = tf.Graph()
  with g.as_default():
    with open(pb, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.compat.v1.import_graph_def(graph_def, name='')
    pool3 = g.get_tensor_by_name('pool_3:0')
    ops = g.get_operations()
    for op_idx, op in enumerate(ops):
        for out in op.outputs:
            ts = out.get_shape()
            if ts.rank is None:
                continue
            new_shape = []
            for j in range(ts.rank):
                dim = ts[j]
                if dim is None:
                    d = None
                else:
                    d = int(dim)
                if d == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(d)
            out.set_shape(tf.TensorShape(new_shape))
    w = g.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)
  _inception_sess = tf.compat.v1.Session(graph=g)


if _inception_sess is None:
  _init_inception()
