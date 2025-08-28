function plotErrorModes_onFDGrid_allSaved(final_lambdas, final_weights, w1, b1, fd_modes, X, Y, domain)
    numModes = length(final_lambdas);
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

        % Plot/save NN mode
        figure;
        surf(X, Y, Z_nn, 'EdgeColor', 'none');
        view(2); axis equal tight;
        shading interp;
        colorbar;
        colormap jet;
        caxis(global_clim);
        xlabel('x'); ylabel('y');
        print(gcf, sprintf('NN_Mode_%d.svg', i), '-dsvg');
        close;

        % Plot/save FD mode
        figure;
        surf(X, Y, Z_fd, 'EdgeColor', 'none');
        view(2); axis equal tight;
        shading interp;
        colorbar;
        colormap jet;
        caxis(global_clim);
        xlabel('x'); ylabel('y');
        print(gcf, sprintf('FD_Mode_%d.svg', i), '-dsvg');
        close;

        % Plot/save error mode 
        figure;
        surf(X, Y, Z_diff, 'EdgeColor', 'none');
        view(2); axis equal tight;
        shading interp;
        colorbar;
        colormap jet;
        caxis(global_clim);
        xlabel('x'); ylabel('y');
        print(gcf, sprintf('Error_Mode_%d.svg', i), '-dsvg');
        close;


    end
end
