function plot_temp(final_lambdas, final_weights, w1, b1, fd_modes, X, Y, domain, loss_convergence, eigen_convergence)

    numModes = min(length(final_lambdas), numel(fd_modes));

    [gridSize, ~] = size(X);
    xy_vec = [X(:)'; Y(:)'];
    mask = domain(:)' == 1;

    % --- Compute global color limits across all modes ---
    allData = [];
    for i = 1:numModes
        Z_fd = fd_modes{i};
        Z_fd = Z_fd / norm(Z_fd(~isnan(Z_fd))); % normalize same as inside loop
        allData = [allData; Z_fd(~isnan(Z_fd))];
    end
    global_clim = [min(allData), max(allData)];
    % ---------------------------------------------------

    % --- Plot modes ---
    for i = 1:numModes
        % NN reconstruction
        w2 = final_weights(:, i);
        phi_valid = cos(w1 * xy_vec(:, mask) + b1);
        z_valid = (w2' * phi_valid)';
        z_valid = z_valid / norm(z_valid);
        Z_nn = nan(gridSize, gridSize);
        Z_nn(mask) = z_valid;

        % FD mode normalization and sign alignment
        Z_fd = fd_modes{i};
        Z_fd = Z_fd / norm(Z_fd(~isnan(Z_fd)));
        dotprod = nansum(Z_nn(:) .* Z_fd(:));
        if dotprod < 0
            Z_fd = -Z_fd;
        end

        % Absolute error
        Z_diff = abs(Z_nn - Z_fd);

        % --- Save NN mode ---
        figure;
        surf(X, Y, Z_nn, 'EdgeColor', 'none');
        view(2); axis equal tight; shading interp;
        colorbar; colormap jet; caxis(global_clim);
        xlabel('x'); ylabel('y');
        print(gcf, sprintf('NN_Mode_%d.svg', i), '-dsvg'); close;

        % --- Save FD mode ---
        figure;
        surf(X, Y, Z_fd, 'EdgeColor', 'none');
        view(2); axis equal tight; shading interp;
        colorbar; colormap jet; caxis(global_clim);
        xlabel('x'); ylabel('y');
        print(gcf, sprintf('FD_Mode_%d.svg', i), '-dsvg'); close;

        % --- Save error mode ---
        figure;
        surf(X, Y, Z_diff, 'EdgeColor', 'none');
        view(2); axis equal tight; shading interp;
        colorbar; colormap jet; caxis(global_clim);
        xlabel('x'); ylabel('y');
        print(gcf, sprintf('Error_Mode_%d.svg', i), '-dsvg'); close;
    end

    % === Convergence trends ===
    % --- Loss convergence: one plot per mode ---
    for k = 1:numModes
        figure;
        semilogy(loss_convergence{k}, 'LineWidth', 1.5);
        grid on;
        xlabel('Iteration', 'FontSize', 12);
        ylabel('Loss (log scale)', 'FontSize', 12);
        title(sprintf('Loss Convergence - Mode %d', k), 'FontSize', 14);
        print(gcf, sprintf('Loss_Convergence_Mode_%d.svg', k), '-dsvg');
        close;
    end

    % --- Eigenvalue convergence: one plot per mode ---
    for k = 1:numModes
        figure;
        plot(eigen_convergence{k}, 'LineWidth', 1.5);
        grid on;
        xlabel('Iteration', 'FontSize', 12);
        ylabel('Eigenvalue Estimate', 'FontSize', 12);
        title(sprintf('Eigenvalue Convergence - Mode %d', k), 'FontSize', 14);
        print(gcf, sprintf('Eigenvalue_Convergence_Mode_%d.svg', k), '-dsvg');
        close;
    end
end
