
% Author: Zhuo Deng
% Date: Feb, 2016

% compute parameters of rectangular 3d bboxes [x y z l w h theta]
% x,y,z is under camera coordinate system
% theta is under tilted coordinate system

function bbox_3d = VectorizeBboxes(gt3D, Rtilt)
   num_box = numel(gt3D);
   bbox_3d = zeros(num_box, 7);
   for i = 1 : num_box
       % 8 vertex under tilt coordinate system
       corners = gt3D{i};
       % convert to camera system
       corners_cam = (pinv(Rtilt) * corners')'; 
       center = mean(corners_cam, 1);
       %% compute box size [L, W ,H] 
       H = norm(corners(1,:) - corners(5,:));
       d12 = norm(corners(1,:) - corners(2,:));
       d14 = norm(corners(1,:) - corners(4,:));
       box_size = zeros(1, 3);
       
       dir_vec = [];
       if d12 > d14
           box_size(1) = d12; 
           box_size(2) = d14;
           dir_vec = corners(1,:) - corners(4,:);
       else
           box_size(1) = d14;
           box_size(2) = d12;
           dir_vec = corners(1,:) - corners(2,:);
       end
       box_size(3) = H;
       
      %% orientation (degree) as perpendidular to the length
         dir_vec_tilt = dir_vec';
         
         % compute angle
         dx = [1, 0, 0] * dir_vec_tilt;
         dz = [0, 0, 1] * dir_vec_tilt;
         % +z-axis is 0, toward +x is (0, 90], and -x is [-90, 0)
         ang =  atan2d(dx, dz);
         if (ang > 90)
            ang = ang - 180; 
         end
         if (ang < -90)
            ang = ang + 180;
         end
         
       %%  summary
       bbox_3d(i, :) = [center box_size ang];
         
   end
end


