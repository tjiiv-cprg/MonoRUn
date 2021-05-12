from ..builder import DIM_CODERS


@DIM_CODERS.register_module()
class MultiClassNormDimCoder(object):

    def __init__(self,
                 target_means=[
                     (3.89, 1.53, 1.62),   # car
                     (0.82, 1.78, 0.63),   # pedestrian
                     (1.77, 1.72, 0.57)],  # cyclist
                 target_stds=[
                     (0.44, 0.14, 0.11),
                     (0.25, 0.13, 0.12),
                     (0.15, 0.10, 0.14)]):
        super(MultiClassNormDimCoder, self).__init__()
        assert len(target_means) == len(target_stds)
        self.target_means = target_means
        self.target_stds = target_stds

    def encode(self, dimensions, labels):
        target_means = dimensions.new_tensor(self.target_means)
        target_stds = dimensions.new_tensor(self.target_stds)
        dim = dimensions.sub(
            target_means[labels]).div(target_stds[labels])
        return dim

    def decode(self, dim, dim_var, labels):
        target_means = dim.new_tensor(self.target_means)[labels]
        target_stds = dim.new_tensor(self.target_stds)[labels]
        dimensions = dim * target_stds + target_means
        if dim_var is not None:
            dimensions_var = dim_var * target_stds.square()
        else:
            dimensions_var = None
        return dimensions, dimensions_var
