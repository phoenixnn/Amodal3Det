function  hlist = draw_box3d(corners, color, linewidth)
% corners: 8 x 3

draw_seq = [1, 2;
            2, 3;
            3, 4;
            4, 1;
            1, 5;
            2, 6;
            3, 7;
            4, 8;
            5, 6;
            6, 7;
            7, 8;
            8, 5];

num_lines = size(draw_seq, 1);
hlist = cell(num_lines, 1);
for i = 1 : size(draw_seq, 1)
   hlist{i} = plot3(corners(draw_seq(i, :), 1), ...
              corners(draw_seq(i, :), 2), ...
              corners(draw_seq(i, :), 3), ...
              color, 'LineWidth', linewidth);
end


end

