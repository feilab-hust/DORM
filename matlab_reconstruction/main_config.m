function main_config(bath_path, mip_num, mean, std, volume_num, gpu_index, mag, doef_data, dl_path)
    t1 = clock;
    bath_path = strrep(bath_path, '\', '/');

    magnification = mag;
    if magnification == 100
        theta = 44/180*pi;
    elseif magnification == 60
        theta = 54/180*pi;
    elseif magnification == 40
        theta = 45/180*pi;
    elseif magnification == 30
        theta = 65/180*pi;
    elseif magnification == 20
        theta = 66/180*pi;
    elseif magnification == 10
        theta = 78/180*pi;
    elseif magnification == 4
        theta = 81/180*pi;
    end
    
    
    pixel_size = 6.5;

    temp = dir(fullfile(bath_path,'v_*'));
    View_name = {temp.name};
    view_num = size(View_name, 2); 
%     lens_distance = pixel_size * magnification*12;  % micro


    layer1_shift_h = 0;   % lens3            layer 1
    layer1_shift_w = 0;  % lens3             layer 1
    layer3_shift_h = 0;  % shift in lens1   layer 3
    layer3_shift_w = 0;  % shift in lens1   layer 3
    
    using_doef_data = doef_data;
    if using_doef_data && (mag == 10 || mag == 20)
        scan_in_lens = 1;
    else
        scan_in_lens = 0;
    end
    for i = 1:view_num
        
        config = split(View_name{i},'_');
        n_split = str2double(config(2));
        view_mip_num = mip_num*n_split;
        x_slice_per_layer = str2double(config(3));
        x_stepsize = str2double(strrep(config(4),'-','.'));
        z_num_per_stack = str2double(config(5));   
        lens_distance = str2double(strrep(config(6),'-','.'));
        n_pixel = round(lens_distance / pixel_size * magnification);
        if scan_in_lens
            n_pixel = round(n_pixel / 2);
        end
        n_over_lab = floor(n_pixel/2);  
        factor = x_stepsize * magnification / pixel_size;
        view_path = fullfile(bath_path,View_name{i});
        run_view(view_path, view_mip_num, mean, std, ...
            n_split, x_slice_per_layer, theta, ...
            factor, volume_num, gpu_index, ...
            n_pixel ,n_over_lab,layer1_shift_h ,layer1_shift_w ...
            ,layer3_shift_h ,layer3_shift_w, using_doef_data, ...
            z_num_per_stack,scan_in_lens,dl_path);
    end
    t2 = clock;
    s = etime(t2,t1);
    m = floor(s/60);
    s = s-m*60;
    h = floor(m/60);
    m = m-h*60;
    fprintf(['总耗时',num2str(h),'h',num2str(m),'m',num2str(s),'s\n'])
 end

    
