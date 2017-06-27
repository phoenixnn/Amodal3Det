% For NYU dataset v2
% convert dmap to [0, 255], where 255 represents 10 meters

data_path = '../../dataset/NYUV2/dmap_f';
for i = 1 : 1449
   load(fullfile(data_path, [num2str(i), '.mat'])); 
   mask = (dmap_f > 10);
   dmap_f(mask) = 10;
   dmap_f = dmap_f/10*255;
   save(fullfile('dmap_f', [num2str(i), '.mat']), 'dmap_f');
end