function corners = calc_corners_tilt(boxes_3d, Rtilt)
  % center point
  center = Rtilt * boxes_3d(1:3)';
  
  % box dimension
  half_L = boxes_3d(4)/2;
  half_W = boxes_3d(5)/2;
  half_H = boxes_3d(6)/2;
  
  % crude
  corners = [ half_L, half_H, half_W;
             -half_L, half_H, half_W;
             -half_L, half_H, -half_W;
              half_L, half_H, -half_W;
              half_L, -half_H, half_W;
             -half_L, -half_H, half_W;
             -half_L, -half_H, -half_W;
              half_L, -half_H, -half_W];
  % rotation       
  theta = boxes_3d(end); % radian
  R = [cos(theta) 0 sin(theta);
        0 1 0;
        -sin(theta) 0 cos(theta)];
  corners = R * corners';
   
  corners = corners' + repmat(center', [8 1]);
end

