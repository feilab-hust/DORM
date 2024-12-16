function configure(config)
    t1 = clock;
    temp = dir(fullfile(config.root,'v_*'));
    View_name = {temp.name};
    view_num = size(View_name, 2); 

    for i = 1:view_num
        
        config_str = split(View_name{i},'_');
        config.slice_per_stack = str2double(config_str(2));
        config.stepsize = str2double(strrep(config_str(3),'-','.'));
        config.zstep = str2double(strrep(config_str(4),'-','.'));
        config.l1_shift = str2double(config_str(5));
        config.l2_shift = str2double(config_str(6));
        
        config.factor = config.stepsize / config.pixelsize;
        config.view_path = fullfile(config.root,View_name{i});
        save_path = fullfile(config.view_path,'reconstructed');
        if ~exist(save_path, 'dir')
            mkdir(save_path);
        end
        config.save_path = save_path;
        
        fprintf([strrep(config.view_path,'\','/'),'\n']);
        step1_entry(config);
    end
    
    t2 = clock;
    dt1 = etime(t2,t1);
    m = floor(dt1/60);
    s = dt1-m*60;
    h = floor(m/60);
    m = m-h*60;
    fprintf(['Total time',num2str(h),'h',num2str(m),'m',num2str(s),'s\n'])

end

