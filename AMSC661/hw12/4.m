function KS_solver()
close all
fsz = 20; % fontsize

N = 512;
L = 32 * pi;
x = (0:N-1) * L / N;
k = [0:N/2-1, -N/2:-1]' * (2*pi / L); % wave numbers

% 初始条件，强制列向量
u0 = cos(x/16) .* (1 + sin(x/16));
u0 = u0(:);

dt = 0.25; % time step
tmax = 200;
nmax = round(tmax / dt);

% 线性算子：-u_xxxx - u_xx
Lk = k.^4 - k.^2;

% 指数算子
E = exp(-dt * Lk);
E2 = exp(-dt * Lk / 2);

% 非线性项微分因子
g = -0.5i * k;

% Fourier 变换初值
v = fft(u0);

% 保存结果矩阵（列：时间步，行：空间）
uu = zeros(N, nmax);
tt = (0:nmax-1) * dt;

% 保存初始状态
uu(:,1) = real(ifft(v));

% 时间积分
for n = 2:nmax
    Nv = g .* fft(real(ifft(v)).^2);
    a = dt * E2 .* Nv;
    va = E .* (v + a/2);
    
    Nv = g .* fft(real(ifft(va)).^2);
    b = dt * E2 .* Nv;
    vb = E .* (v + b/2);
    
    Nv = g .* fft(real(ifft(vb)).^2);
    c = dt * E2 .* Nv;
    vc = E .* (v + c);
    
    Nv = g .* fft(real(ifft(vc)).^2);
    d = dt * E2 .* Nv;
    
    v = E .* v + (a + 2*b + 2*c + d) / 6;
    
    % 保存当前结果，强制列向量
    uu(:, n) = real(ifft(v(:)));
end

% 绘制结果
figure;
imagesc(tt, x, uu);
axis xy; colormap(jet);
xlabel('Time'); ylabel('x');
title('Kuramoto-Sivashinsky equation');
colorbar;
end
