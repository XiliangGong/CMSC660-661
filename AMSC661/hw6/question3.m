% 图1: L形状网格生成
figure(1)
fd = @(p) ddiff(drectangle(p, -1.5, 1.5, -1.5, 1.5), drectangle(p, 0, 1.5, 0, 1.5));
% 定义固定点
fixed_points = [-1.5, -1.5; -1.5, 1.5; 1.5, -1.5; 1.5, 0; 0, 1.5; 0, 0];
[p, t] = distmesh2d(fd, @huniform, 0.1, [-1.5, -1.5; 1.5, 1.5], fixed_points);
title('L-shape Mesh with Adjusted Grid Size');

% 图2: 五边形与圆形网格生成
figure(2)
% 定义五边形的角度和固定点
phi = (0:5)' / 5 * 2 * pi + pi / 10;
pfix1 = 1.5 * [cos(phi), sin(phi)];  
r = sin(pi / 5) / sind(108 / 2) * cosd(72) / cosd(108 / 2);
phi = (0:5)' / 5 * 2 * pi - pi / 10;
pfix2 = 1.5 * r * [cos(phi), sin(phi)];  
% 定义多边形区域
fd = @(p) ddiff(dpoly(p, pfix1), dpoly(p, pfix2));
[p, t] = distmesh2d(fd, @huniform, 0.12, [-1.5, -1.5; 1.5, 1.5], [pfix1; pfix2]);
title('Pentagon and Circle Mesh with Adjusted Grid Size');

% 图3: 圆形与两个小圆形网格生成
figure(3)
fd = @(p) ddiff(max(sqrt(sum(p.^2, 2)) - 1.5, p(:, 2)), sqrt((p(:, 1) + 0.75).^2 + (p(:, 2) + 0.6).^2) - 0.375);
fd = @(p) ddiff(fd(p), sqrt((p(:, 1) - 0.75).^2 + (p(:, 2) + 0.6).^2) - 0.375);
[p, t] = distmesh2d(fd, @huniform, 0.05, [-1.5, -1.5; 1.5, 1.5], [-1.5, 0; 1.5, 0]);
title('Circle with Two Small Circles Mesh with Adjusted Grid Size');


