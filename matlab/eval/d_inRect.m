function In = d_inRect(xy, rect)
n_pts = size(xy, 1);

% to homogenours coordinates
xy = [xy, ones(n_pts, 1)];

dist = zeros(n_pts, 4);
coor = [4, 1;
        1, 2;
        2, 3;
        3, 4];

for i = 1 : 4
    n = rect(coor(i,1), :) - rect(coor(i, 2), :);
    c = -dot(n, rect(i, :));
    dist(:, i) = dist2line([n, c], xy);
end

In = (dist(:,1) > 0) & (dist(:,2) >0) & (dist(:,3) > 0) & (dist(:,4) > 0);

end

function dist = dist2line(line, pts)
    n_pts = size(pts, 1);
    line = repmat(line, n_pts, 1);
    dist = sum(line.*pts, 2);
end

