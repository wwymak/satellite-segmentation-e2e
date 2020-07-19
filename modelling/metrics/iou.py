from ignite.metrics import Metric

import segmentation_models_pytorch as smp

class IoU(Metric):

    def __init__(self, output_transform=lambda x: x):

        self.sum_iou = None
        self.batch_count = None
        super().__init__(output_transform)

    def reset(self):
        self.sum_iou = 0
        self.batch_count = 0

    def update(self, output):
        y_pred, y = output
        print(y_pred.shape, y.shape)
        batch_iou = smp.utils.metrics.IoU()(y_pred.squeeze(), y.squeeze())
        print(batch_iou, 'batch)iou')
        self.sum_iou += batch_iou
        self.batch_count +=1
        # ... your custom implementation to update internal state on after a single iteration
        # e.g. self._var2 += y.shape[0]

    def compute(self):
        # compute the metric using the internal variables
        # res = self._var1 / self._var2
        print(self.batch_count, self.sum_iou)
        res = self.sum_iou/self.batch_count
        return res