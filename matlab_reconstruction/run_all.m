function timepoint = run_all(data_path, view_path,  mip_num, mean, std, ...
                             n_split, n_slice_per_stack, theta, factor, ...
                             data_type, data_count, volume_num, gpu_index, ...
                             n_pixel ,n_over_lab,layer1_shift_h ,layer1_shift_w ...
                             ,layer3_shift_h ,layer3_shift_w,using_doef_data, ...
                             z_num_per_stack,H_name, scan_in_lens,dl_path)
    
    if data_count>=mip_num  
        timepoint = 0;   % 
        return
    end
    mip_num = mip_num - data_count;

    
    if gpu_index
        gpuDevice(gpu_index);
    end

    temp = dir(fullfile(data_path,'*.tif'));  
    data_name = {temp.name};
    data_num = size(data_name, 2); 
    stack_size_list = zeros(data_num,1);
    for i = 1:data_num
        temp = imfinfo(fullfile(data_path, data_name{i}));
        stack_size_list(i) = size(temp,1);   
        depth = sum(stack_size_list);  
        if depth>mip_num*n_slice_per_stack
            break
        end
    end
    info = temp(1);
    height   = info.Height;
    width    = info.Width;
    bitdepth = info.BitDepth;
    
    depth = sum(stack_size_list);  
    
    layer_num = floor(depth / n_slice_per_stack); 
    timepoint = floor(layer_num / z_num_per_stack); 
    timepoint = min([timepoint,mip_num]);  



    for i = 1:timepoint
        for n = 1:layer_num
            tic;
            fprintf([num2str(i*n),'\t\t']);   
            fprintf('t%d_layer%d\t\t\n', i, n);  
            
            temp = floor((i*n-1)/volume_num);  
            volum_number = mod(temp,n_split);  
            save_rootdir = fullfile(view_path,sprintf('\\Reconstruction_%d\\',volum_number));   
            if ~exist(save_rootdir, 'dir')
                mkdir(save_rootdir)
            end
    
            shift_matrix = get_shift(data_type, factor, theta);  
            correction_matrix = get_correction(data_type, factor, theta);  
    
            a = (i*n-1)*n_slice_per_stack+1;  
            b = i *n* n_slice_per_stack;
            name_list = (a:b);
            if using_doef_data   
                if scan_in_lens
                    imgArray = {};
                    for s = 1:2
                        for l = 1:size(H_name, 2)
                            data_path = fullfile(view_path, H_name{l});
                            img_l = read_stack(data_path, data_name, stack_size_list, name_list, data_type, height, width, bitdepth);
                            stack = img_l;
                            if data_type
                                stack_size = size(stack);
                                stack_size(2) = round(stack_size(2)/cos(theta));
                                stack = imresize3(stack, stack_size);
                            else
                                stack_size = size(stack);
                                stack_size(1) = round(stack_size(1)/cos(theta));
                                stack = imresize3(stack, stack_size);
                            end        
                            
                            for j = 1:size(stack,4)
                                if gpu_index
                                    G = gpuArray(uint16(stack(:,:,:,j)));
                                else
                                    G = uint16(stack(:,:,:,j));
                                end
                        
                                result = imwarp(G,affine3d(shift_matrix), 'linear');         % 'cubic','nearest'，'linear'
                                
                                zf = factor*sin(theta);
                                result_size = size(result);
                                result_size(3) = round(result_size(3)*zf);
                                result = imresize3(result, result_size);
                                
                
                                result = imwarp(result,affine3d(correction_matrix), 'linear');
                                
                                depth = ceil(size(result, 3)/2);
                                if data_type
                                    depth_goal = ceil(size(stack,2)/2*sin(theta));
                                else
                                    depth_goal = ceil(size(stack,1)/2*sin(theta));
                                end
                                result = result(:,:,depth-depth_goal:depth+depth_goal);
                                
                                if mean>0 && mean>3*std
                                    noise = uint16(randn(size(result))*std + mean);
                                    flag_ = result<mean-3*std;
                                    result(flag_) = noise(flag_);
                                end                
                                result=permute( result,[2,1,3]);
                            end  
                            imgArray{end+1} = result;
                        end
                    end
                        [img1_1, img2_1, img3_1,img1_2, img2_2, img3_2] = deal(imgArray{:});   
                        [h,w,d_r] = size(img1_1);

                        out1_1 = crop_and_shift_rotate(img3_1,d_r, n_pixel, mean,n_over_lab ,layer1_shift_w, layer1_shift_h, -0.25,0.25);
                        out1_2 = crop_and_shift_rotate(img3_2,d_r, n_pixel, mean,n_over_lab ,layer1_shift_w, layer1_shift_h, -0.25,0.25);
                        out2_1 = crop_and_shift_rotate(img2_1,d_r, n_pixel, mean,n_over_lab ,0, 0, 0,0);
                        out2_2 = crop_and_shift_rotate(img2_2,d_r, n_pixel, mean,n_over_lab ,0, 0, 0,0);
                        out3_1 = crop_and_shift_rotate(img1_1,d_r, n_pixel, mean,n_over_lab ,layer3_shift_w, layer3_shift_h, 0,0);
                        out3_2 = crop_and_shift_rotate(img1_2,d_r, n_pixel, mean,n_over_lab ,layer3_shift_w, layer3_shift_h, 0,0);
                             
                        result = zeros(h, w, 6*n_pixel + n_over_lab, 'uint16');
                        result(:, :, 1:size(out1_1, 3)) = out1;
 
                        overlap_index1_2 = n_pixel;   %
                        result = stitching(result,out1_2, n_pixel, overlap_index1_2,n_over_lab);
                        overlap_index2_1 = 2*n_pixel;
                        result = stitching(result,out2_1, n_pixel, overlap_index2_1,n_over_lab);
                        overlap_index2_2 = 3*n_pixel;
                        result = stitching(result,out2_2, n_pixel, overlap_index2_2,n_over_lab);
                        overlap_index3_1 = 4*n_pixel;
                        result = stitching(result,out3_1, n_pixel, overlap_index3_1,n_over_lab);
                        overlap_index3_2 = 5*n_pixel;
                        result = stitching(result,out3_2, n_pixel, overlap_index3_2,n_over_lab);

                else
                    imgArray = {};
                    for l = 1:size(H_name, 2)                       
                        data_path = fullfile(view_path, H_name{l});
                        img_l = read_stack(data_path, data_name, stack_size_list, name_list, data_type, height, width, bitdepth);
                        stack = img_l;
                        if data_type
                            stack_size = size(stack);
                            stack_size(2) = round(stack_size(2)/cos(theta));
                            stack = imresize3(stack, stack_size);
                        else
                            stack_size = size(stack);
                            stack_size(1) = round(stack_size(1)/cos(theta));
                            stack = imresize3(stack, stack_size);
                        end        
                        
                        for j = 1:size(stack,4)
                            if gpu_index
                                G = gpuArray(uint16(stack(:,:,:,j)));
                            else
                                G = uint16(stack(:,:,:,j));
                            end
                    
                            result = imwarp(G,affine3d(shift_matrix), 'linear');         % 'cubic','nearest'，'linear'
                            
                            zf = factor*sin(theta);
                            result_size = size(result);
                            result_size(3) = round(result_size(3)*zf);
                            result = imresize3(result, result_size);
                            
            
                            result = imwarp(result,affine3d(correction_matrix), 'linear');
                            
                            depth = ceil(size(result, 3)/2);
                            if data_type
                                depth_goal = ceil(size(stack,2)/2*sin(theta));
                            else
                                depth_goal = ceil(size(stack,1)/2*sin(theta));
                            end
                            result = result(:,:,depth-depth_goal:depth+depth_goal);
                            
                            if mean>0 && mean>3*std
                                noise = uint16(randn(size(result))*std + mean);
                                flag_ = result<mean-3*std;
                                result(flag_) = noise(flag_);
                            end                
                            result=permute( result,[2,1,3]);
                        end  
                        imgArray{end+1} = result;
                    end
                    [img1, img2, img3] = deal(imgArray{:});
            
                    [h,w,d_r] = size(img1);
                    n_pixel2 = n_pixel;
                    n_pixel1 = n_pixel;

                    out1 = crop_and_shift_rotate(img3,d_r, n_pixel1, mean,n_over_lab ,layer1_shift_w, layer1_shift_h, -0.25,0.25);
                    out2 = crop_and_shift_rotate(img2,d_r, n_pixel2, mean,n_over_lab ,0, 0, 0,0);
                    out3 = crop_and_shift_rotate(img1,d_r, n_pixel1, mean,n_over_lab ,layer3_shift_w, layer3_shift_h, 0,0);

                    %stitching lens data     
                    result = zeros(h, w, n_pixel2 + n_pixel1 * 2 + n_over_lab, 'uint16');
                    result(:, :, 1:size(out1, 3)) = out1;
                    overlap_index1 = n_pixel1;   %
                    result = stitching(result,out2, n_pixel1, overlap_index1,n_over_lab);
                    overlap_index2 = 2*n_pixel1;
                    result = stitching(result,out3, n_pixel2, overlap_index2,n_over_lab);
                end
            else
                stack = read_stack(data_path, data_name, stack_size_list, name_list, data_type, height, width, bitdepth);
                for j = 1:size(stack,4)            
        
                    if gpu_index
                        G = gpuArray(uint16(stack(:,:,:,j)));
                    else
                        G = uint16(stack(:,:,:,j));
                    end
           
                    result = imwarp(G,affine3d(shift_matrix), 'linear');         % 'cubic','nearest'，'linear'
                    
                    zf = factor*sin(theta);
                    result_size = size(result);
                    result_size(3) = round(result_size(3)*zf);
                    result = imresize3(result, result_size);
                    
                    result = imwarp(result,affine3d(correction_matrix), 'linear');
                    
                    depth = ceil(size(result, 3)/2);
                    if data_type
                        depth_goal = ceil(size(stack,2)/2*sin(theta));
                    else
                        depth_goal = ceil(size(stack,1)/2*sin(theta));
                    end
                    result = result(:,:,depth-depth_goal:depth+depth_goal);
                    
                    if mean>0 && mean>3*std
                        noise = uint16(randn(size(result))*std + mean);
                        flag_ = result<mean-3*std;
                        result(flag_) = noise(flag_);
                    end                
                    result=permute( result,[2,1,3]);
                    d_r = size(result,3);
                    c1 = ceil(d_r/2);
                    d_s = floor(c1 - (n_pixel + n_over_lab) / 2);
                    d_e = d_s + n_pixel + n_over_lab-1;
                    result = result(:, :, d_s:d_e);
                end
            end
            dt = toc;
            fprintf([num2str(dt), '\n']);
            if n == 1
                [h,w,d_s] = size(result);
                z_step = d_s-n_over_lab;
                result_sti = zeros(h, w, layer_num*z_step + n_over_lab, 'uint16');
                result_sti(:, :, 1:size(result, 3)) = result;
            else
                overlap_index_s = z_step*(n-1);   %
                result_sti = stitching(result_sti,result, z_step, overlap_index_s,n_over_lab);
            end
        end

%     sti_dir = fullfile(save_rootdir, '3D_stitching');
    sti_dir = save_rootdir;
    if ~exist(sti_dir, 'dir')
        mkdir(sti_dir)
    end

    result_sti = result_sti(20:end-20,:,:);
    for m = 1:size(result_sti, 3)
        if m == 1
            imwrite(result_sti(:,:,m), fullfile(sti_dir, sprintf('%05d.tif', i)));
        else
            imwrite(result_sti(:,:,m), fullfile(sti_dir, sprintf('%05d.tif', i)),'WriteMode','append');
        end
    end
    %     sti_dir_mip = fullfile(save_rootdir, 'MIPxy_3D_stitching');
%     if ~exist(sti_dir_mip, 'dir')
%         mkdir(sti_dir_mip)
%     end
%     
%     result_mip = max(result_sti, [], 3);
%     imwrite(result_mip, fullfile(sti_dir_mip, sprintf('%05d.tif', i)));

    rootPath = mfilename('fullpath');
    parentDir = fileparts(fileparts(rootPath));
    targetDir = fullfile(parentDir, 'code', 'options', 'test');
    % yamlFilePath = fullfile(targetDir, 'test_CropPatch.yml');
    yamlFilePath_bat = fullfile(targetDir, 'test_CropPatch_bat.yml');
    ck_dir_new = fullfile(parentDir, 'experiments', ...
    '20241208_60x_Digital_DOFe_ds6_and_ds5_mask_6p5_good_lens2', ...
    'models', 'best_C.pth');
    data_dir_new = sti_dir;

     if isfile(yamlFilePath_bat)
        yamlData = ReadYaml(yamlFilePath_bat);
        yamlData.network_C.n_fea = int32(yamlData.network_C.n_fea);
        yamlData.val.crop_size_d = int32(yamlData.val.crop_size_d);
        yamlData.val.crop_size_h = int32(yamlData.val.crop_size_h);
        yamlData.val.crop_size_w = int32(yamlData.val.crop_size_w);
        yamlData.val.min_devide_by = int32(yamlData.val.min_devide_by);
        yamlData.val.over_lap = int32(yamlData.val.over_lap);
        ck_dir_old = yamlData.path.pretrain_model_C;
        data_dir_old = yamlData.datasets.test_1.dataroot_LQ;

        WriteYaml(yamlFilePath_bat, yamlData);
        disp('Updated YAML Data written successfully.');

    else
        error('YAML file not found: %s', yamlFilePath);
    end

    fileContent = fileread(yamlFilePath_bat);
    fileContent = strrep(fileContent,  data_dir_old, data_dir_new);
    fileContent = strrep(fileContent, ck_dir_old, ck_dir_new);
    
    fid = fopen(yamlFilePath_bat, 'w');
    fwrite(fid, fileContent);
    fclose(fid);

    dl_code = fullfile(parentDir, 'code', 'inference_bat.py');
    
    % cmd = 'C:\ProgramData\Anaconda3\envs\torch\python.exe J:\clb\Cell_up_load_data\sort_up_load20241129\DORM1205\code\inference_bat.py ';
    dl_python = fullfile(dl_path, 'python.exe');
    cmd = sprintf('"%s" "%s"', dl_python, dl_code);
    system(cmd);

    % result_mip = max(result_sti, [], 3);
    % imwrite(result_mip, fullfile(sti_dir_mip, sprintf('%05d.tif', i)));
    end
end

function out = crop_and_shift_rotate(img,d_r, n_pixel, mean,n_over_lab ,shift_w,  shift_h, xy_rotate,xz_rate)
        c1 = ceil(d_r/2);
        d_s1 = floor(c1 - (n_pixel + n_over_lab) / 2);
        d_e1 = d_s1 + n_pixel + n_over_lab-1;
        out = img(:, :, d_s1:d_e1);
        if shift_w >0 || shift_h>0
            out = imtranslate(out, [shift_w, shift_h, 0], 'FillValues', mean);
        end
        if xy_rotate ~= 0
            out = imrotate3(out,xy_rotate,[0 1 0],'cubic','crop');   
        end
        if xz_rate ~= 0
            out = imrotate3(out, xz_rate,[0 0 1],'cubic','crop');   
        end
end

function result = stitching(result,img, n_pixel, overlap_index,n_over_lab)
        previous_data = result(:, :, overlap_index+1:overlap_index + n_over_lab);
        current_data = img(:, :, 1:n_over_lab);
        for j = 1:n_over_lab
            current_weight = j / (n_over_lab + 1);
            previous_weight = 1 - current_weight;
            result(:, :, overlap_index + j) = uint16(previous_weight * previous_data(:, :, j) + ...
                                                     current_weight * current_data(:, :, j));
        end
        result_copy_index = overlap_index + n_over_lab;
        result(:, :, result_copy_index+1:result_copy_index + n_pixel) = img(:, :, n_over_lab+1:end);
end

