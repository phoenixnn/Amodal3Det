
function boxes_3d = get_3d_proposals( dmap, K, sp_base, ...
                                    sp_label, depth_fill, avg_boxDims_cls)
  
   % xyz camera coordinates
   xyz = Dmap2PointCloud(dmap, K);
   xyz = reshape(xyz, [], 3);
   d_f = depth_fill(:);
   
   num_cls = size(avg_boxDims_cls, 1) + 1;
   num_props = numel(sp_label);
   centers = zeros(num_props, 3);
   boxes_3d = zeros(num_props, 7*num_cls);
   for i = 1: num_props
       % centroid
       mask_ids = get_superpixel_mask(sp_base, sp_label{i});
       xyz_seg = xyz(mask_ids, :);
       
       % remove missing data
       nz_ids = xyz_seg(:, 3) ~= 0;
       xyz_seg_nz = xyz_seg(nz_ids, :);
       
       if isempty(xyz_seg_nz)
           tmp = d_f(mask_ids);
           tmp(tmp==0) = [];
           assert(~isempty(tmp))
           cz = median(tmp);
       else
           cz = median(xyz_seg_nz(:, 3));
       end
       assert(cz ~= 0);
  
       % 2d center
       [r, c] = ind2sub(size(sp_base), mask_ids);
       x = mean(c); y = mean(r);
       cx = (x-K(1,3))*cz/K(1,1);  
       cy = (y-K(2,3))*cz/K(2,2);
       centers(i, :) = [cx, cy, cz];
   end
   
   
   for i = 2 : num_cls
       start = (i-1)*7 + 1; 
       boxes_3d(:, start : (start+2) ) = centers;
       boxes_3d(:, (start+3) : (start+5)) = ...
                  repmat(avg_boxDims_cls(i-1, :), num_props, 1);
   end
  
end

function mask_ids = get_superpixel_mask(sp_base, sp_label)
     mask = ismember(sp_base, sp_label);
     mask_ids = find(mask);
end


