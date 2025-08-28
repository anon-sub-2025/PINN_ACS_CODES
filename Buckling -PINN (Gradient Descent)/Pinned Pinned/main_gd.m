clc;
clear;
close all;
tic;
rng(42);
n = 500;             % Number of neurons
L = 1;               % Length of the rod

W_1 = dlarray(2 * rand(n, 1) - 1);
b = dlarray(2 * rand(n, 1) - 1);
w_2 = dlarray(2 * rand(n, 1) - 1);
lambda = dlarray(1);

m2 = 50;
delta = 0.01;
x_colloc = linspace(delta, L - delta, m2);
theoriticalEigenVal = 9.869 ;
% Optimizer storage
averageGrad = [];
averageSqGrad = [];

numIterations=12000;
lossHistory = zeros(numIterations, 1); 
eigenValues =[];



for iter = 1:numIterations

    [loss, gradW_1, gradb, gradw_2, grad_lambda] = dlfeval(@(W_1,b,w_2,lambda) ...
        modelLoss(W_1, b, w_2, lambda, x_colloc, m2, n, L), W_1, b, w_2, lambda);

    params = [W_1; b; w_2; lambda];
    grads = [gradW_1; gradb; gradw_2; grad_lambda];

    [params, averageGrad, averageSqGrad] = adamupdate(params, grads, averageGrad, averageSqGrad, iter,4*1e-3);

    W_1 = params(1:n);
    b = params(n+1:2*n);
    w_2 = params(2*n+1:3*n);
    lambda = params(end);
    eigenValues = [eigenValues , lambda];
    % Save loss
    lossHistory(iter) = double(gather(extractdata(loss)));

    fprintf('Iteration = %d, Loss = %.6f, Lambda = %.6f\n', ...
        iter, lossHistory(iter), double(gather(extractdata(lambda))));
end
time_elapsed = toc;
fprintf("time elapsed %.4f \n" , time_elapsed);
%% === PLOTTING SECTION (with PDF export) ===
set(groot,'defaultAxesFontName','Times New Roman', ...
    'defaultAxesFontSize',14, ...
    'defaultLineLineWidth',1.8, ...
    'defaultLineMarkerSize',6);

% ----- Loss vs Iteration -----
fig1 = figure;
semilogy(1:numIterations, lossHistory, 'b','LineWidth',2);
xlabel('Iteration','FontWeight','bold');
ylabel('Loss (log scale)','FontWeight','bold');
title('Training Loss','FontWeight','bold');
grid on; box on;
exportgraphics(fig1,'loss_history.pdf','ContentType','vector');  % PDF

% ----- Eigenvalue Convergence -----
fig2 = figure;
plot(1:numIterations, eigenValues, 'b','LineWidth',2); hold on;
yline(theoriticalEigenVal,'r--','LineWidth',2, ...
    'DisplayName','Theoretical Value');
xlabel('Iteration','FontWeight','bold');
ylabel('Eigenvalue','FontWeight','bold');
title('Eigenvalue Convergence','FontWeight','bold');
legend('Estimated','Theoretical','Location','best');
grid on; box on;
exportgraphics(fig2,'eigenvalue_convergence.pdf','ContentType','vector');  % PDF

% ----- Predicted Mode Shape -----
x_vec = 0:0.01:L;
ypred = w_2' * cos(W_1*x_vec+ b);
coeff = sqrt(sum(ypred.^2));
ypred = ypred/coeff; % L2 normalization

fig3 = figure;
plot(x_vec , ypred , 'r','LineWidth',2);
xlabel('x','FontWeight','bold');
ylabel('y(x)','FontWeight','bold');
title('Predicted Mode Shape','FontWeight','bold');
grid on; box on;
exportgraphics(fig3,'mode_shape.pdf','ContentType','vector');  % PDF
function [loss, gradW_1, gradb, gradw_2, grad_lambda] = modelLoss(W_1,b,w_2,lambda ,x_colloc , m2 ,n, L)

    YPDE = zeros(m2, 1);
    d4w_dx4 = (cos(W_1*x_colloc + b).*(W_1(:,1).^4))';
    d2w_dx2 = -(cos(W_1*x_colloc + b) .*(W_1(:,1).^2))' ;
    res = d4w_dx4 +lambda*d2w_dx2;
    % Boundary conditions for pinned-pinned beam
    x_bc = zeros(4, n);
    y_bc = zeros(4, 1);
    x_bc(1, :) = cos(W_1 * 0 + b)'; % w(0)
    x_bc(2, :) = cos(W_1 * L + b)'; %w(L))
    x_bc(3, :) = -(cos(W_1 * 0 + b) .* (W_1(:, 1).^2))'; % w''(0)
    x_bc(4, :) = -(cos(W_1 * L + b) .* (W_1(:, 1).^2))'; % w''(L)
    % sample some reference points to avoid trivial solution
    xd = 0.5;
    Xref = cos(W_1 * xd + b)';
    Yref = ones(size(Xref,1), 1);
    w_pde = 1e-5;
    w_bc =1;
    w_ref =1;
    pdeLoss = w_pde *(1/(size(res,1))) *sum((res*w_2-YPDE).^2) ;
    bc_loss = w_bc*(1/(size(x_bc,1))) *sum((x_bc*w_2-y_bc).^2) ;
    ref_loss = w_ref* sum((Xref*w_2-Yref).^2) ;
    loss =  pdeLoss + bc_loss + ref_loss ;

    % Compute gradients w.r.t. parameters
    [gradw_2, grad_lambda, gradW_1, gradb] = dlgradient(loss, w_2, lambda, W_1, b);

end

