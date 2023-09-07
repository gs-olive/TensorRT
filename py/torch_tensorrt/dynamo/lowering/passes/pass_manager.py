from torch.fx.passes.pass_manager import PassManager


class DynamoPassManager(PassManager):
    def __init__(self, passes=None, constraints=None):
        super().__init__(passes, constraints)

    @classmethod
    def build_from_passlist(cls, passes):
        pm = DynamoPassManager(passes)
        return pm

    def __call__(self, gm, sample_inputs):
        self.validate()
        out, example_inputs = gm, sample_inputs
        for _pass in self.passes:
            out = _pass(out, example_inputs)
        return out
