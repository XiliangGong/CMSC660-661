function [vert, tria] = generate_halfcircle_with_holes_mesh(N, R, hole_radius, hfun)
%GENERATE_HALFCIRCLE_WITH_HOLES_MESH Generate a mesh of a lower half circle with two holes.
%


    if nargin < 1, N = 100; end
    if nargin < 2, R = 1.5; end
    if nargin < 3, hole_radius = 0.375; end
    if nargin < 4, hfun = 0.1; end

    node = [];
    edge = [];

    for n = 1:N
        theta = -pi * n / N;  % from -pi to 0 (downward half circle)
        x = R * cos(theta);
        y = R * sin(theta);
        node = [node; x, y];
        edge = [edge; n, n + 1];
    end
    edge(end, 2) = 1;  % close the half-circle loop

    cx1 = -0.75; cy1 = -0.6;
    for n = 1:N
        theta = -2 * pi * n / N;  % clockwise
        x = cx1 + hole_radius * cos(theta);
        y = cy1 + hole_radius * sin(theta);
        node = [node; x, y];
        edge = [edge; N + n, mod(n, N) + N + 1];
    end

    cx2 = 0.75; cy2 = -0.6;
    for n = 1:N
        theta = -2 * pi * n / N;  % clockwise
        x = cx2 + hole_radius * cos(theta);
        y = cy2 + hole_radius * sin(theta);
        node = [node; x, y];
        edge = [edge; 2 * N + n, mod(n, N) + 2 * N + 1];
    end

    [vert, ~, tria, ~] = refine2(node, edge, [], [], hfun);

    figure;
    patch('faces', tria(:, 1:3), 'vertices', vert, ...
          'facecolor', 'w', 'edgecolor', [0.2, 0.2, 0.2]);
    hold on; axis image off;
    patch('faces', edge(:, 1:2), 'vertices', node, ...
          'facecolor', 'w', 'edgecolor', [0.1, 0.1, 0.1], 'linewidth', 1.5);
    set(gcf, 'units', 'normalized', 'position', [0.05, 0.50, 0.30, 0.35]);
    drawnow;

end
