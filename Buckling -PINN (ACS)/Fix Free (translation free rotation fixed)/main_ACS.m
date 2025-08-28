% Differential equation: d⁴w/dx⁴ + λ² * d²w/dx² = 0, where λ² = P / (EI)
% Assuming rod of length 1
clc;
clear;
close all;
tic;
rng(42);
n = 500;             % Number of neurons
L = 1;               % Length of the rod

% Bandwidth [-1, 1]
w1 = (2 * rand(n, 1) - 1);
b1 = (2 * rand(n, 1) - 1);

% sample some reference points to avoid trivial solution
xd = rand(1 ,3);

Xref = cos(w1 * xd + b1)';
Yref = ones(size(Xref,1), 1);

% Generating collocation points for PDE constraint
m2 = 50;
delta = 0.01;
x_colloc = linspace(delta, L - delta, m2);
YPDE = zeros(m2, 1);

d4w_dx4 = (cos(w1*x_colloc + b1).*(w1(:,1).^4))';
d2w_dx2 = -(cos(w1*x_colloc + b1) .*(w1(:,1).^2))' ;

% === Boundary Conditions for Fixed-free (rotational fixed - translational free) Beam ===
x_bc = zeros(3, n);
y_bc = zeros(3, 1);
x_bc(1, :) = cos(w1 * 0 + b1)';                            % w(0) = 0
x_bc(2, :) = (-sin(w1 * 0 + b1) .* w1)';                  % w'(0) = 0
x_bc(3, :) = (-sin(w1 * L + b1) .* w1)';                  % w'(L) = 0
theoriticalEigenVal = 9.869;
A = d4w_dx4;
B = -d2w_dx2;
%  adding the eigen value depended boundary condition % w'''(L) + λ² w'(L) 
A(end+1 ,:) =(sin(w1 * L + b1).* (w1.^3) )';
B(end+1 ,:) = -(-sin(w1*L+ b1).* w1(:,1))'; 
numIterations =6000;
tol =1e-8;

eigenVal =1; % starting Guess
w2Opt = inf; 
lossHistory=[];
eigenValues =[];
%% biconvex optimisation (for finding critical load -first eigen value)
for iter = 1:numIterations  
    w2 = inf; % store best w2 found so far
    minLoss = inf ; % store best Loss found so far 
    fprintf("iteration %d\n" ,iter); 
    res = A - (eigenVal) * B;
    eigenValues =[eigenValues ,eigenVal];
    %  to avoid nodal points 
    for j =1: size(Xref , 1) 

        X = [res; 1e2*x_bc; Xref( j, :)];         
        Y = [zeros(m2+1,1); 1e2*y_bc; Yref(j)];      

        wopt = pinv(X) * Y;
        loss = norm(X*wopt - Y) ;
        if(loss < minLoss)
            minLoss =loss  ;
            w2 = wopt ;
        end

    end 
    lossHistory =[lossHistory ,loss];
    num = (((A*w2)') * (B*w2))  + (((B*w2)') *(A*w2)) ;
    den = 2* (((B*w2)')*(B*w2)) ;
    newApproxEigen = ((num)/(den)) ;

    % if abs(newApproxEigen - eigenVal) < tol
    %     break;
    % end
    eigenVal = (newApproxEigen);
    w2Opt = w2;
end
elapsed_time =toc;
fprintf("crtical buckling load (fix-free(translational free-rotational fix)) %.6f\n " ,eigenVal);
fprintf("time elapsed %.4f\n" , elapsed_time);
%% === PLOTTING SECTION (with PDF export) ===
set(groot,'defaultAxesFontName','Times New Roman', ...
    'defaultAxesFontSize',14, ...
    'defaultLineLineWidth',1.8, ...
    'defaultLineMarkerSize',6, ...
    'defaultAxesToolbarVisible','off');   % turn off toolbar

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
ypred = w2Opt' * cos(w1*x_vec+ b1);
coeff = sqrt(sum(ypred.^2));
ypred = ypred/coeff; % L2 normalization

fig3 = figure;
plot(x_vec , ypred , 'r','LineWidth',2);
xlabel('x','FontWeight','bold');
ylabel('y(x)','FontWeight','bold');
title('Predicted Mode Shape','FontWeight','bold');
grid on; box on;
exportgraphics(fig3,'mode_shape.pdf','ContentType','vector');  % PDF
