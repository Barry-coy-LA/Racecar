clc, clf, clear all, close all

% 读取 CSV 文件
data = readtable('simulation_data.csv');

% 将表格中的列提取为向量
laptime = data.Laptime;
speed = data.Speed;
steering_angle = data.SteeringAngle;
steering = data.Steering;
theta = data.Theta;
right_distance = data.RightDistance;
desired_distance = data.DesiredDistance;
error_term = desired_distance - 0.7 * right_distance;

figure;
plot(laptime, speed, 'LineWidth', 1.5);
xlabel('Laptime (s)');
ylabel('Speed (m/s)');
title('Speed vs Laptime');

figure;
plot(laptime, rad2deg(steering), 'LineWidth', 1.5);
hold on
plot(laptime, rad2deg(steering_angle), 'LineWidth', 1.5);
plot(laptime, rad2deg(theta),'LineWidth',1.5)
y1 = 19*ones(1,length(theta));
y2 = -19*ones(1,length(theta));
plot(laptime, y1,'k--', 'LineWidth', 1.5)
plot(laptime, y2,'k--', 'LineWidth', 1.5)
hold off
xlabel('Laptime (s)');
ylabel('steer (degree)');
legend('Steering Angle','Steering Angle (Constraint)','Theta','Steering Angle constraint')
title('Steering vs Laptime');

figure;
plot(laptime, desired_distance, 'LineWidth', 1.5);
hold on
plot(laptime, 0.7*right_distance, 'LineWidth', 1.5);
plot(laptime, error_term, 'LineWidth', 1.5);
y3 = zeros(1,length(error_term));
plot(laptime, y3, 'k--','LineWidth', 1.5)
hold off
xlabel('Laptime (s)');
ylabel('distance (m/s)');
legend('Desired distance','Right distance','Error term')
title('Distance vs Laptime');

% % 定义 x 轴的范围
% x1 = -19:0.0005: 19;
% x11 = deg2rad(x1);
% y1 = 8.25 * (0.002.^abs(x11));
% x = steering_angle(800:1600);  % 从 0 到 5，生成 100 个点
% t = laptime(800:1600);
% 
% % 计算 y = exp(x) 的值
% y = 8.25 *(0.002.^abs(x));

% % 创建图形窗口
% figure;
% hold on;
% plot(x1, y1, 'b-', 'LineWidth', 2);  % 绘制指数函数曲线
% 
% % 初始化点
% 
% h = plot(rad2deg(x(1)), y(1), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'k');  % 初始点，红色
% hxline = plot([x(1), x(1)], [1, y(1)], 'k--');  % 垂直虚线，连接到 x 轴
% hyline = plot([-20, x(1)], [y(1), y(1)], 'k--');  % 水平虚线，连接到 y 轴
% 
% x_mid = mean(x1);  % x 轴的中点
% y_top = max(y1) + 0.04 * max(y1);  % y 轴上方 10% 处
% 
% txt = text(x_mid, y_top, sprintf('x = %.2f, y = %.2f,  = %.2f', rad2deg(x(1)), y(1), t(1)), ...
%     'FontSize', 16, 'HorizontalAlignment', 'center', 'BackgroundColor', 'none');
% 
% xlabel('Steeing Angle');
% ylabel('Speed');
% title('Speed change');
% grid on;
% % 创建动画
% for k = 1:length(x)
%     % 更新点的位置
%     set(h, 'XData', rad2deg(x(k)), 'YData', y(k));
%     set(hxline, 'XData', [rad2deg(x(k)), rad2deg(x(k))], 'YData', [1, y(k)]);
%     set(hyline, 'XData', [-20, rad2deg(x(k))], 'YData', [y(k), y(k)]);
%     set(txt, 'String', sprintf('Steering Angle = %.2f degree, Speed = %.2f,\n laptime = %.2f s', rad2deg(x(k)), y(k), t(k)));
% 
%     % 捕获当前帧
%     frame = getframe(gcf);
%     im = frame2im(frame);
%     [imind, cm] = rgb2ind(im, 256);
% 
%     % 短暂停留，控制动画速度
%     pause(0.01);
% end
% 
% % 添加标题和标签
% 
% hold off;
% 
% % 定义 x 轴的范围
% x = linspace(-5, 5, 1000);  % 从 -5 到 5，生成 1000 个点
% 
% % 计算 y = tanh(x) 的值
% y = 19*tanh(x);
% 
% % 创建图形窗口并绘制 tanh 函数
% figure;
% plot(x, y, 'b-', 'LineWidth', 2);  % 绘制蓝色实线，线宽为 2
% 
% % 添加网格线
% grid on;
% 
% % 设置图像标题和坐标轴标签
% title('y = tanh(x)');
% xlabel('x');
% ylabel('y');
% 
% % 设置 x 轴和 y 轴的
% 
% % 定义 x 轴的范围
% x = linspace(0, 2, 1000);  % 从 0 到 2，生成 1000 个点
% 
% % 计算不同底数小于1的指数函数
% y1 = 0.0001.^x;   % 底数为 0.5 的指数函数
% y2 = 0.001.^x;   % 底数为 0.8 的指数函数
% y3 = 0.01.^x;   % 底数为 0.3 的指数函数
% y4 = 0.1.^x;   % 底数为 0.9 的指数函数
% 
% % 创建图形窗口并绘制不同底数的指数函数
% figure;
% hold on;
% plot(x, y1, 'r-', 'LineWidth', 2);  % 红色实线，底数为 0.5
% plot(x, y2, 'g-', 'LineWidth', 2);  % 绿色实线，底数为 0.8
% plot(x, y3, 'b-', 'LineWidth', 2);  % 蓝色实线，底数为 0.3
% plot(x, y4, 'm-', 'LineWidth', 2);  % 紫色实线，底数为 0.9
% hold off;
% 
% % 添加图例
% legend('0.0001^x', '0.001^x', '0.01^x', '0.1^x', 'Location', 'northeast');
% 
% % 添加网格线
% grid on;
% 
% % 设置图像标题和坐标轴标签
% title('Exponential Transformation');
% xlabel('x');
% ylabel('y');
% 
% % 设置 x 轴和 y 轴的范围
% xlim([0 2]);
% ylim([0 1]);
