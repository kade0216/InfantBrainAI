def create_model(opt):

    # if opt.model == 'local3D':
    #     from .PTN_model3D import PTN
    #     model = PTN()
    if opt.model == '3DUNet':
        from .model import UNet3D
        model = UNet3D(in_channels=1, out_channels=opt.cls_num, final_sigmoid=False, is_segmentation=False,f_maps=32)
    elif opt.model == '3DResUNet':
        from .model import ResidualUNet3D
        model = ResidualUNet3D(in_channels=1, out_channels=opt.cls_num, final_sigmoid=False, is_segmentation=False)
    return model
