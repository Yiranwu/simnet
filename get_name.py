def get_raw_dataset_name(config):
    return '%d-%dx%d'%(config.start_template_id, config.end_template_id, config.num_mods)

def get_dataset_name(config):
    return '%d-%dx%d%s%s'%(config.start_template_id, config.end_template_id, config.num_mods,
                            '_pad' if config.padding else '',
                            '_corr' if config.corrupt else '')

def get_exp_name(config):
    return '%s_%sep%d_lr%f_h%d_%s'%(config.model_name,'ev_' if config.eval else 'noev_',
                                      config.epochs, config.lr,
                                      config.hidden, config.dataset_name)
