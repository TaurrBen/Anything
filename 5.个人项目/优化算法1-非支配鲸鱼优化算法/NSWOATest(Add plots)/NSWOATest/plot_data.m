function plot_data(M,D,Pareto,ParetoMean)
% This function to plot Pareto solution
pl_data= Pareto(:,D+1:D+M); % extract data to plot
% n = size(pl_data,1);
% k=1;
% for i = 1:n
%     if (pl_data(i,1)<76) && (pl_data(i,2)<98845.74) && (pl_data(i,3)<264657.99)
%         ParetoBest(k,:)=Pareto(i,:);
%         k=1+k;
%     end
% end
ParetoBest=Pareto((pl_data(:,1)<76) ...
    & (pl_data(:,2)<98845.74) ...
    & (pl_data(:,3)<264657.99),:);

pl_data2=ParetoBest(:,D+1:D+M); % extract data to plot
pl_data2=sortrows(pl_data2,3);
X=pl_data2(:,1);
Y=pl_data2(:,2);
Z=pl_data2(:,3);
save ParetoBest.txt ParetoBest -ascii;  % save data for future use
writematrix(ParetoBest,'ParetoBest.xlsx');
n=length(Z);
c=linspace(0,100,n);
figure;
scatter3(X, Y, Z,30,c,'o', 'filled');
view(-30, 20);
% Add title and axis labels
title('Optimal Solution Pareto Set');
xlabel('Objective function T');
ylabel('Objective function E');
zlabel('Objective function C');

name = ['T','E','C'];
for i = 1:M
    figure;
    plot(ParetoMean(:,i));
    title(sprintf('Average of %s',name(i)));
    xlabel('Number of iteration');
    ylabel(sprintf('Average of %s',name(i)));
end
end