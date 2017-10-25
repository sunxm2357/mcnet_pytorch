
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'mcnet':
        from .mcnet_model import McnetModel
        model = McnetModel()
    else:
        raise ValueError("Model [%s] not recognized."% opt.model)

    model.initialize(opt)
    print("model [%s] is created" % (model.name()))

    return model