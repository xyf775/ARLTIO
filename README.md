# ARLTIO
function [X2, success, info] = MRD_DDM_from_data(V, I, a1, a2, Rs, T_K, bounds, opts)
    success=true;
    info=struct();
    info.cond_history=[]; info.lambda_history=[]; info.res_history=[]; info.relchg=[]; info.colnorms=[];

    q=1.602176634e-19; kB=1.380649e-23;
    V=V(:); I_meas=I(:); N=numel(V);
    safeexp=@(x) exp(min(max(x,-700),700));

    % detect Rp bounds -> convert to theta bounds if needed
    if ~isempty(bounds) && size(bounds,1)>=4
        if bounds(4,1) > 1e-8 && bounds(4,2) > bounds(4,1)
            theta_low = 1 / bounds(4,2);
            theta_high = 1 / bounds(4,1);
            bounds(4,1)=theta_low; bounds(4,2)=theta_high;
        end
    else
        bounds = [0,1; 0,1e-3; 0,1e-3; 1e-6, 1];
    end

    Iph0 = max(max(I_meas), 0.9*mean(I_meas));
    I01_0 = 1e-7;
    I02_0 = 1e-7;
    theta0 = 1 / max(1e-3, mean([bounds(4,1), bounds(4,2)]));
    x_raw = [Iph0; I01_0; I02_0; theta0];
    I_curr = I_meas;

    for iter=1:opts.max_iter
        % build A for double diode model
        Vk = V + I_curr .* Rs;
        Vt = kB * T_K / q;
        expo1 = Vk ./ (a1 * Vt);
        expo2 = Vk ./ (a2 * Vt);
        expo1 = min(max(expo1,-700),700);
        expo2 = min(max(expo2,-700),700);
        Gi1 = -(safeexp(expo1)-1);
        Gi2 = -(safeexp(expo2)-1);
        Ki = -Vk;
        A = [ones(N,1), Gi1, Gi2, Ki];
        A(~isfinite(A))=0; I_meas(~isfinite(I_meas))=0;

        % column scaling
        colnorms = sqrt(sum(A.^2,1))';
        colnorms(colnorms < eps) = 1;
        Scol = diag(1./colnorms);
        A_scaled = A * Scol;
        info.colnorms(:,end+1)=colnorms;

        % IRLS weight (Huber-like)
        if iter==1
            w = ones(N,1);
        else
            r_prev = info.res_history(end,:)';
            delta = opts.huber_delta;
            absr = abs(r_prev);
            huber_w = ones(N,1);
            idx = absr > delta; huber_w(idx) = delta ./ absr(idx);
            w = huber_w ./ (sqrt(r_prev.^2 + opts.eps_w^2));
            w(~isfinite(w))=1; w = min(max(w,1e-3),1e3);
        end
        W_sqrt = diag(sqrt(w));
        Aw = W_sqrt * A_scaled;
        bw = W_sqrt * I_meas;
        Aw(~isfinite(Aw))=0; bw(~isfinite(bw))=0;

        % solve (QR then SVD+Tikhonov)
        lambda = 0;
        try
            if opts.use_qr
                [Q,R] = qr(Aw,0);
                if all(isfinite(R(:))), sR = svd(R); condR = sR(1)/max(sR(end),eps); else condR=Inf; end
                info.cond_history(end+1)=condR;
                if condR < 1/opts.inner_tol
                    z = R \ (Q' * bw);
                else
                    [U,S,Vv] = svd(Aw,'econ'); s = diag(S);
                    smax = max(s); smin = max(min(s), eps);
                    condA = smax / smin;
                    lambda = opts.lambda0 * max(1, condA^opts.alpha);
                    S_inv_reg = diag(s ./ (s.^2 + lambda.^2));
                    z = Vv * S_inv_reg * (U' * bw);
                    info.cond_history(end+1) = condA;
                end
            else
                [U,S,Vv] = svd(Aw,'econ'); s = diag(S);
                smax = max(s); smin = max(min(s), eps);
                condA = smax / smin;
                lambda = opts.lambda0 * max(1, condA^opts.alpha);
                S_inv_reg = diag(s ./ (s.^2 + lambda.^2));
                z = Vv * S_inv_reg * (U' * bw);
                info.cond_history(end+1) = condA;
            end
        catch
            Aw(~isfinite(Aw))=0;
            [U,S,Vv] = svd(Aw,'econ'); s = diag(S);
            smax = max(s); smin = max(min(s), eps);
            condA = smax / smin;
            lambda = opts.lambda0 * max(1, condA^opts.alpha);
            S_inv_reg = diag(s ./ (s.^2 + lambda.^2));
            z = Vv * S_inv_reg * (U' * bw);
            info.cond_history(end+1) = condA;
        end

        x_new = Scol * z;

        if ~isempty(bounds)
            Iph_c = max(min(x_new(1), bounds(1,2)), bounds(1,1));
            I01_c = max(min(x_new(2), bounds(2,2)), bounds(2,1));
            I02_c = max(min(x_new(3), bounds(3,2)), bounds(3,1));
            theta_c = max(min(x_new(4), bounds(4,2)), bounds(4,1));
            x_new = [Iph_c; I01_c; I02_c; theta_c];
        end

        resid = A * x_new - I_meas;
        resid(~isfinite(resid))=0;
        info.res_history(end+1,:) = resid';
        relchg = norm(x_new - x_raw) / max(norm(x_raw), eps);
        info.relchg(end+1) = relchg;
        info.lambda_history(end+1) = lambda;

        x_raw = x_new;

        % update I_curr self-consistently with adaptive relaxation
        Iph = x_raw(1); I01 = x_raw(2); I02 = x_raw(3); theta = x_raw(4);
        Rp = 1 / max(theta, 1e-12);
        Vd = V + I_curr .* Rs;
        Vt = kB * T_K / q;
        expo1_model = Vd ./ (a1 * Vt);
        expo2_model = Vd ./ (a2 * Vt);
        expo1_model = min(max(expo1_model,-700),700);
        expo2_model = min(max(expo2_model,-700),700);
        I_model = Iph - I01 .* (safeexp(expo1_model) - 1) - I02 .* (safeexp(expo2_model) - 1) - Vd ./ Rp;
        alpha_relax = max(0.4, min(0.9, 1 - 0.3 * exp(-iter/3)));
        I_curr = alpha_relax * I_model + (1 - alpha_relax) * I_curr;
        I_curr(~isfinite(I_curr)) = I_meas(~isfinite(I_curr));

        if relchg < opts.inner_tol, break; end
    end

    Iph = x_raw(1); I01 = x_raw(2); I02 = x_raw(3); theta = x_raw(4);
    Rp = 1 / max(theta, 1e-12);
    X2 = [Iph; I01; I02; Rp];
    success = all(isfinite(X2));
    if ~isempty(info.cond_history), info.condA_final = info.cond_history(end); end
end
