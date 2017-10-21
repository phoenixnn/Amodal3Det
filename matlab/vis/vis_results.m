% An example of visualizing colorized point cloud and 3d bounding box

%% load annotation file
var = load('dataset/NYUV2/annotations/1.mat');
data = var.data;
im = data.img;
Rtilt = data.Rtilt;
K = data.K;
rawDepth = data.rawDepth;

% recover 3d points
xyz = Rgbd2PointCloud(im, rawDepth, K);

% draw pcd 
f = figure;
ax = pcshow(xyz, im); 
grid off;
set(gca, 'Visible','off');
set(gca,'color','none');
hold on;

% draw box
corners_bb3d = data.gt3D{10};
corners_bb3d = (pinv(Rtilt)*corners_bb3d')';
h = draw_box3d(corners_bb3d, 'b', 2);
