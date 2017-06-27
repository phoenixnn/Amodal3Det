% 

function IoU = calc_IoU_3d(corners1, corners2, v1, v2)
% corners: 8 x 3
y1_min = min(corners1(:,2));
y1_max = max(corners1(:,2));
y2_min = min(corners2(:,2));
y2_max = max(corners2(:,2));

% no overlap
if (y1_min >= y2_max) || (y1_max <= y2_min)
    IoU = 0;
    return
end

% height
height = min(y2_max, y1_max) - max(y2_min, y1_min);

% bottom area
rect1 = corners1(1:4, [1, 3]);
rect2 = corners2(1:4, [1, 3]);
area = calc_intersect_rect_fast(rect1, rect2);

v = height*area;
IoU = v/(v1 + v2 - v);

end

