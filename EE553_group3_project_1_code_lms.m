% EE553 Group 3 Project 1
% Vehicle Vibration Error Compensation Using Accelerometer Data 
% and LMS Algorithm
% 6/12/26
% Carson Forsyth
% Anthony Scott, EI
% Tom Sellers
clear all; close all; clc;

%% ------------------------------------------------------------------------
%  Parameters
%  ------------------------------------------------------------------------
fs = 2000;       % Sampling Frequency (2kHz)
t = (0:1/fs:5)';  % 5 Second Time Vector
N = length(t);   % Number of Samples

%% ------------------------------------------------------------------------
%  Signal Generation
%  ------------------------------------------------------------------------
% Desired Signal: Vehicle Acceleration
s = 2 * sin( 2 * pi * 0.5 * t);  % Used 0.5Hz for low frequency acceleration on a car

% Predictable Noise (Engine ticking): Used 300Hz and 600Hz
v_engine = 0.5 * sin(2 * pi * 300 * t) + 0.25 * sin(2 * pi * 600 * t);

% Unpredictable Noise (Road): Used Band-Limited at 150Hz
raw_road = randn(N, 1);
v_road = 0.8 * lowpass(raw_road, 150, fs);

% Input d(n): Corrupted accelerometer
d = s + v_engine + v_road;

% Reference Input x(n)
x = 0.6 * sin(2 * pi * 300 * t + pi / 6) + ...
    0.3 * sin(2 * pi * 600 * t + pi / 8) + ...
    0.5 * lowpass(raw_road, 150, fs);

%% ------------------------------------------------------------------------
%  Least Mean Squares (LMS)
%  ------------------------------------------------------------------------
filter_length_list = [16 32 64 128 256]; % Number of Filter weights (taps)
mu_list = [0.05 0.075 0.1 0.125 0.15]; % Step size
mse_table = zeros(length(filter_length_list), length(mu_list));

for fl_idx = 1 : numel(filter_length_list)
    for m_idx = 1 : numel(mu_list)
        filter_length = filter_length_list(fl_idx);
        mu = mu_list(m_idx);

        w = zeros(filter_length, 1);  % Weights vector
        y = zeros(N, 1);              % Estimated Noise Output
        e = zeros(N, 1);              % Error

        w_mags = zeros(N, 1); % Track magnitude of filter weights
        
        % Adaptive Filter Loop
        for i = filter_length : N
            % Pull the current window of the reference signal
            x_vec = x(i : -1 : i - filter_length + 1);
        
            % Calculate filter output (Estimated Noise)
            new_y = w' * x_vec;
            if isnan(new_y) || isinf(new_y)
                break
            end
            y(i) = new_y;
        
            % Calculated error
            new_e = d(i) - y(i);
            if isnan(new_e) || isinf(new_e)
                break
            end
            e(i) = new_e;
        
            % Update Weights
            w = w + mu * e(i) * x_vec;

            % Track weight vector magnitude
            w_mags(i) = norm(w');
        end

        %% ------------------------------------------------------------------------
        %  Results
        %  ------------------------------------------------------------------------
        figure(1);
        
        % Corrupted Signal
        subplot( ...
            2 * numel(filter_length_list), ...
            numel(mu_list), ...
            2 * (fl_idx - 1) * numel(filter_length_list) + m_idx ...
        );
        plot(t, d, 'r');
        hold on;
        plot(t, s, 'k', 'LineWidth', 1.5);
        grid on;
        xlim([0, 5]);
        
        % Cleaned Signal
        subplot( ...
            2 * numel(filter_length_list), ...
            numel(mu_list), ...
            (2 * (fl_idx - 1) + 1) * numel(filter_length_list) + m_idx ...
        );
        plot(t(100 : end), e(100 : end), 'r');
        hold on;
        plot(t(100 : end), s(100 : end), 'k', 'LineWidth', 1.5);
        grid on;
        xlim([0, 5]);

        % Used Welch's Method to estimate the power spectra
        window = hanning(1024);
        noverlap = 512;
        nfft = 2048;

        [Pxx_d, F_d] = pwelch(d, window, noverlap, nfft, fs);
        [Pxx_e, F_e] = pwelch(e, window, noverlap, nfft, fs);

        % Plotted PSD in dB/Hz
        figure(2);
        subplot( ...
            numel(filter_length_list), ...
            numel(mu_list), ...
            (fl_idx - 1) * numel(filter_length_list) + m_idx ...
        );
        plot(F_d, 10 * log10(Pxx_d), 'r', 'LineWidth', 1.2);
        hold on;
        plot(F_e, 10 * log10(Pxx_e), 'b', 'LineWidth', 1.2);
        grid on;
        xlim([0 1000]);

        % Plot filter weight vector magnitude over time
        figure(3);
        subplot( ...
            numel(filter_length_list), ...
            numel(mu_list), ...
            (fl_idx - 1) * numel(filter_length_list) + m_idx ...
        );
        plot(1 : N, w_mags);

        % Calculate MSE
        mse_score = 0;
        for i = filter_length : N
            % Pull the current window of the reference signal
            x_vec = x(i : -1 : i - filter_length + 1);

            % Calculated error
            error = d(i) - w' * x_vec;
            if isnan(new_e) || isinf(new_e)
                break
            end
            mse_score = mse_score + (error ^ 2 - mse_score) / i;
        end
        mse_table(fl_idx, m_idx) = mse_score;
    end
end

figure(1);
sgtitle('Primary Sensor Input and LMS Output');
han = axes(gcf, 'visible', 'off');
han.XLabel.Visible = 'on';
han.YLabel.Visible = 'on';
xlabel(han, 'Time (seconds)');
ylabel(han, 'Amplitude (m/s^2)');

figure(2);
sgtitle('Power Spectral Density: Noise Cancellation Profile');
han = axes(gcf, 'visible', 'off');
han.XLabel.Visible = 'on';
han.YLabel.Visible = 'on';
xlabel(han, 'Frequency (Hz)');
ylabel(han, 'Power/Frequency (dB/Hz)');

figure(3);
sgtitle('Adaptive Filter Weight Magnitude');
han = axes(gcf, 'visible', 'off');
han.XLabel.Visible = 'on';
han.YLabel.Visible = 'on';
xlabel(han, 'Step');
ylabel(han, 'Weight Magnitude');

% Display MSE table
mse_fig = figure(4);
uitable(mse_fig, ...
    'Data', mse_table, ...
    'ColumnName', mu_list, ...
    'RowName', filter_length_list ...
);
