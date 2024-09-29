clc ; clf; clear all; close all

data = readtable('output_PurePursuit.csv');
data1 = readtable('output_Stanley.csv');

% 将表格中的列提取为向量
laptime = data.Var1;
speed = data.Var2;
steering_angle = data.Var3;
Lateral_Error = data.Var4;
mean_Lateral_Error = mean(Lateral_Error);  % 计算y的均值
rms_Lateral_Error = sqrt(mean(Lateral_Error.^2));  % 计算y的均方根

laptime1 = data1.Var1;
speed1 = data1.Var2;
steering_angle1 = data1.Var3;
Lateral_Error1 = data1.Var4;
mean_Lateral_Error1 = mean(Lateral_Error1);  % 计算y的均值
rms_Lateral_Error1 = sqrt(mean(Lateral_Error1.^2));  % 计算y的均方根

figure;
plot(laptime, speed, 'LineWidth', 1.5);
hold on
plot(laptime1, speed1, 'LineWidth', 1.5);
xlabel('Laptime (s)');
ylabel('Speed (m/s)');
legend('PurePursuit','Stanley')
title('Speed vs Laptime');

figure;
plot(laptime, Lateral_Error, 'LineWidth', 1.5);
hold on
plot(laptime1, Lateral_Error1, 'LineWidth', 1.5);
plot([laptime(1), laptime(end)], [mean_Lateral_Error, mean_Lateral_Error], '--', 'LineWidth', 1.5);
plot([laptime(1), laptime(end)], [mean_Lateral_Error1, mean_Lateral_Error1], '--', 'LineWidth', 1.5);

x_center = mean([laptime(1), laptime(end)]);  % x轴中心
y_top = max(Lateral_Error);  % y轴最上方的稍上位置

text(x_center, y_top, sprintf('RMS(PP) = %.2f', rms_Lateral_Error), ...
     'FontSize', 12);  
text(x_center, y_top-0.1, sprintf('RMS(Stanley) = %.2f', rms_Lateral_Error1), ...
     'FontSize', 12);

xlabel('Laptime (s)');
ylabel('Speed (m/s)');
legend('PurePursuit','Stanley','PurePursuit mean','Stanley mean')
title('Lateral Error vs Laptime');

figure;
plot(laptime, steering_angle, 'LineWidth', 1.5);
hold on
plot(laptime1, steering_angle1, 'LineWidth', 1.5);
xlabel('Laptime (s)');
ylabel('steering_angle (rad)');
legend('PurePursuit','Stanley')
title('steering angle vs Laptime');


img = imread('Silverstone_map.png');

data2 = readtable('Silverstone_map_info.csv');
data3 = readtable('Silverstone_map_QP.csv');

center_x = data2.x_X_m;
center_y = data2.y_m;
% center_wtr_r = data2.w_tr_right_m;
% center_wtr_l = data2.w_tr_left_m;

qp_x = data3.x_X_m;
qp_y = data3.y_m;

x_scaled = (center_x / max(center_x)) * size(img, 2);  % 将x缩放到图像宽度范围
y_scaled = (center_y - min(center_y)) / (max(center_y) - min(center_y)) * size(img, 1);  % 将y缩放到图像高度范围
y_scaled = size(img, 1) - y_scaled;

x_scaled_qp = (qp_x / max(qp_x)) * size(img, 2);  % 将x缩放到图像宽度范围
y_scaled_qp = (qp_y - min(qp_y)) / (max(qp_y) - min(qp_y)) * size(img, 1);  % 将y缩放到图像高度范围
y_scaled_qp = size(img, 1) - y_scaled_qp;

figure;
% imshow(img); 
hold on;
axis on;
axis image;  % 确保坐标系与图像尺寸匹配

plot(x_scaled, y_scaled, 'r-', 'LineWidth', 2);
plot(x_scaled_qp, y_scaled_qp, 'k-', 'LineWidth', 2);

title('Curve on PNG Image');
xlabel('X axis');
ylabel('Y axis');
hold off;

data4 = readtable('Silverstone_map_calculation.csv');
sm = data4.s_m;
velocity = data4.v_mps;
acceleration = data4.a_mps2;

figure
yyaxis left
plot(sm, velocity, 'b-', 'LineWidth', 2);  % Plot speed with blue line
ylabel('Speed (km/h)');  % Label for speed axis
xlabel('Distance (m)');  % Label for x-axis
grid on;

% Plot the right y-axis for acceleration
yyaxis right
plot(sm, acceleration, 'r--', 'LineWidth', 2);  % Plot acceleration with red dashed line
ylabel('Acceleration (m/s^2)');  % Label for acceleration axis

% Add title
title('Speed and Acceleration vs Distance');

% Optional: Add legend
legend('Speed', 'Acceleration', 'Location', 'best');
