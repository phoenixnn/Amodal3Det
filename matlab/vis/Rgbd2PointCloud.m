function xyz = Rgbd2PointCloud(rgb, rawDepth, K)
% compute point cloud under camera system

% xyz: mxnx3
[h, w, ~] = size(rgb);
cx = K(1,3); cy = K(2,3);  
fx = K(1,1); fy = K(2,2);
[x, y] = meshgrid(1:w, 1:h);   
x3 = (x-cx).*rawDepth/fx;  
y3 = (y-cy).*rawDepth/fy;
z3 = rawDepth;
xyz = cat(3, x3, y3, z3);

end

