%% 拟合: '43.97*x^-1.212+1.603'。


pc2surfac emesh()
[xData, yData] = prepareCurveData( dis, fx );

% 设置 fittype 和选项。
ft = fittype( 'power2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [2265.00441090766 -0.0527557109234848 -0.860955382211273];

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, opts );

% 绘制数据拟合图。
figure( 'Name', '44210*x^-1.214+1603' );
h = plot( fitresult, xData, yData );
legend( h, '主距fx', '44210*x^-1.214+1603', 'Location', 'NorthEast', 'Interpreter', 'none' ,'FontSize',15);
% 为坐标区加标签
xlim([0 1200]);
xlabel( '标定板距离', 'Interpreter', 'none' ,'FontSize',18);
ylabel( '主距fx', 'Interpreter', 'none' ,'FontSize',18);

%% 拟合: '43.97*x^-1.212+1.603'。
[xData, yData] = prepareCurveData( dis, fy );

% 设置 fittype 和选项。
ft = fittype( 'power2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [2261.03738269336 -0.0525191052911172 -0.845126375048181];

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, opts );

% 绘制数据拟合图。
figure( 'Name', '41340*x^-1.199+1602' );
h = plot( fitresult, xData, yData );
legend( h, '主距fy', '41340*x^-1.199+1602', 'Location', 'NorthEast', 'Interpreter', 'none' ,'FontSize',15);
% 为坐标区加标签
xlim([0,1200]);
xlabel( '标定板距离', 'Interpreter', 'none' ,'FontSize',18);
ylabel( '主距fy', 'Interpreter', 'none' ,'FontSize',18);


%% 拟合: '43.97*x^-1.212+1.603'。
[xData, yData] = prepareCurveData( dis, k1 );

% 设置 fittype 和选项。
ft = fittype( 'power2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [1.99968177001218e-09 2.5921035729626 0.00668024015345308];

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, opts );

% 绘制数据拟合图。
figure( 'Name', '4.516*x^-0.3203+0.6318' );
h = plot( fitresult, xData, yData );
legend( h, '畸变系数k1', '4.516*x^-0.3203+0.6318', 'Location', 'NorthWest', 'Interpreter', 'none' ,'FontSize',13);
% 为坐标区加标签
xlim([0 1200]);
xlabel( '标定板距离', 'Interpreter', 'none' ,'FontSize',18);
ylabel( '畸变系数k1', 'Interpreter', 'none' ,'FontSize',18);

%% 拟合: '43.97*x^-1.212+1.603'。
[xData, yData] = prepareCurveData( dis, k2 );

% 设置 fittype 和选项。
ft = fittype( 'poly1' );

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, 'Normalize', 'on' );

% 绘制数据拟合图。
figure( 'Name', '-5.86*x-5.993' );
h = plot( fitresult, xData, yData );
legend( h, '畸变系数k2', '-5.86*x-5.993', 'Location', 'NorthEast', 'Interpreter', 'none' ,'FontSize',15);
% 为坐标区加标签-------------------------------------------------------
xlim([0 1200]);
xlabel( '标定板距离', 'Interpreter', 'none' ,'FontSize',18);
ylabel( '畸变系数k2', 'Interpreter', 'none' ,'FontSize',18);


% plot(dis,fx,'*r');plot(linspace(0,1100),43.97*(power(linspace(0,1100),-1.212))+1.603,'r');
% hold on;
% plot(dis,fy,'*b');plot(linspace(0,1100),41.11*(power(linspace(0,1100),-1.198))+1.602,'b');
% xlim([0,1200]);legend(['fx';'fy']);xlabel('标定板距离');ylabel('主距');


%%
bre = [13764977, 15518528, 19589246, 23633528, 30251622, 45297820, 58731721, 90684809, 93371467,90584561, 56458465, 45214568, 29845615, 24114685, 18882541, 14954554, 13999999];
eog = [7682050, 8749321, 10965938, 13226226, 16762488, 25490804, 34967064, 67189403, 67488508, 65445745, 34445841, 25791970, 16190919, 13448541,10888544, 8888455, 7554052];
rob = [14781485, 16857207, 21250302, 25703866, 32676872, 49763467, 68219115, 131365055, 131916708, 131355044, 65471568, 45763879, 32027254, 24986248, 21131983, 15703610, 1475753];
lap = [8.620243773745658, 9.551298132222223, 10.442912391649307, 11.999537224375, 14.908032693333334, 28.50188325305555, 53.8224038016493, 192.59898538530817, 185.06910608155812, 170.295653037482644, 59.321352679474828, 28.74002416134983, 16.075335664166666, 12.7786131781206596, 11.381485097912328, 8.40217019998264, 6.056224924375001];
smd = [1605875.0, 1680353.0, 1746793.0, 1798843.0, 1853988.0, 1929128.0, 1992172.0, 2097132.0, 2147568.0, 1987899.0, 1899656.0, 1838924.0, 1747735.0, 1681792.0, 1540730.0, 1413559.0, 1333130.0];
smd2 = [1291231.0, 1411775.0, 1550984.0, 1697026.0, 1960670.0, 2655202.0, 3326236.0, 4395714.0, 4502671.0, 4139626.0, 3473975.0, 2895380.0, 1862691.0, 1678242.0, 1595867.0, 1463201.0, 1256784.0];

bre = (bre -min(bre))/(max(bre)-min(bre));
eog = (eog -min(eog))/(max(eog)-min(eog));
rob = (rob -min(rob))/(max(rob)-min(rob));
lap = (lap -min(lap))/(max(lap)-min(lap));
smd = (smd -min(smd))/(max(smd)-min(smd));
smd2 = (smd2 -min(smd2))/(max(smd2)-min(smd2));

subplot(2,3,1);
plot(linspace(0,1,17),bre,'*r');
title('Brenner函数',"FontSize",14);
subplot(2,3,2);
plot(linspace(0,1,17),eog,'*r');
title('能量梯度函数',"FontSize",14);
subplot(2,3,3);
plot(linspace(0,1,17),rob,'*r');
title('Roberts函数',"FontSize",14);
subplot(2,3,4);
plot(linspace(0,1,17),lap,'*r');
title('Laplace算子',"FontSize",14);
subplot(2,3,5);
plot(linspace(0,1,17),smd,'*r');
title('灰度方差函数',"FontSize",14);
subplot(2,3,6);
plot(linspace(0,1,17),smd2,'*r');
title('灰度方差乘积函数',"FontSize",14);