clear all
close all
clc
%% Author
% Author      : Federico Giai Pron
% Mail        : federico.giaipron@gmail.com
% Avilability : support, projects / thesis development, script /
% codes / controls writing, etc.
% Experience   : automotive, controls, modelling, finite element method,
% optimization, etc.
%% Input
% Strategy
% 1: 1D minimization test case
% 2: 2D minimization test case
% 3: 1D maximization test case
% 4: 2D maximization test case
tic
Data.Function = 3;
% Problem size
switch Data.Function
    case 1
        Sett.Type    = 'min';
        Sett.LengthX = 1;
    case 2
        Sett.Type    = 'min';
        Sett.LengthX = 2;
    case 3
        Sett.Type    = 'max';
        Sett.LengthX = 318;
    case 4
        Sett.Type    = 'max';
        Sett.LengthX = 2;
end
% Optimization variables limits
% ���߱�������
XLim(1,:) = 1*ones(1,Sett.LengthX);
XLim(2,:) = 7*ones(1,Sett.LengthX);
% Settings
Sett.NumPop    = 5;%Ⱥ����
Sett.NumChr    = 15;%Ⱦɫ����
Sett.NumIter   = 800;%������
Sett.FlagPlots = true;
Sett.Fprintf = false;
%% Run GA
[Results.Xpbest,Results.ObjFunpbest,Results.ObjFuncbestPlot,Results.Scorepbest,Data] = Optimization_GA_v01(XLim, Sett, Data);
toc
%% Plot
% run('MainPlot.m');