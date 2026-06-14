% EE553 Group 3 Project 1
% Vehicle Vibration Error Compensation Using Accelerometer Data 
% and LMS Algorithm
% 6/9/26
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
s = 2 * sin(2*pi*0.5*t);  % Used 0.5Hz for low frequency acceleration on a car

% Predictable Noise (Engine ticking): Used 300Hz and 600Hz
v_engine = 0.5 * sin(2*pi*300*t) + 0.25 * sin(2*pi*600*t);

% Unpredictable Noise (Road): Used Band-Limited at 150Hz
raw_road = randn(N, 1);
v_road = 0.8 * lowpass(raw_road, 150, fs);  % Changed randn to raw_road for Comparsions

% Input d(n): Corrupted accelerometer
d = s + v_engine + v_road;

% Reference Input x(n)
x = 0.6 * sin(2*pi*300*t + pi/6) + ...
    0.3 * sin(2*pi*600*t + pi/8) + ...
    0.5 * lowpass(raw_road, 150, fs);

%% ------------------------------------------------------------------------
%  Normalized LMS (NLMS)
%  ------------------------------------------------------------------------
filter_length = 64;  % Number of Filter weights (taps)
mu = 0.1;            % Normalized step size (0 < mu < 1)
epsilon = 1e-6;      % Small input to avoid divide-by-zero error (BOOM!)

w = zeros(filter_length, 1);  % Weights vector
y = zeros(N, 1);              % Estimated Noise Output
e = zeros(N, 1);              % Error

% Adaptive Filter Loop
for i = filter_length:N
    % Pull the current window of the reference signal
    x_vec = x(i:-1:i-filter_length+1);

    % Calculate filter output (Estimated Noise)
    y(i) = w' * x_vec;

    % Calculated error
    e(i) = d(i) - y(i);

    % Update Weights
    w = w + (mu / (x_vec' * x_vec + epsilon)) * e(i) * x_vec;
end

%% ------------------------------------------------------------------------
%  Results
%  ------------------------------------------------------------------------
figure(1);

% Corrupted Signal
subplot(2,1,1);
plot(t, d, 'r');
hold on;
plot(t, s, 'k', 'LineWidth', 1.5);
title('Primary Sensor Input: Corrupted Accelerometer Data');
xlabel('Time (seconds)');
ylabel('Amplitude (m/s^2)');
legend('Noisy Input d(n)', 'True Acceleration s(n)', 'Location', 'best');
grid on;
xlim([0, 5]);

% Cleaned Signal
subplot(2,1,2);
plot(t(100:end), e(100:end), 'r');
hold on;
plot(t(100:end), s(100:end), 'k', 'LineWidth', 1.5);
title('NLMS Output: Error-Compensated Acceleration');
xlabel('Time (seconds)');
ylabel('Amplitude (m/s^2)');
legend('Cleaned Output e(n)', 'True Acceleration s(n)', 'Location', 'best');
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
plot(F_d, 10*log10(Pxx_d), 'r', 'LineWidth', 1.2);
hold on;
plot(F_e, 10*log10(Pxx_e), 'b', 'LineWidth', 1.2);
title('Power Spectral Density: Noise Cancellation Profile');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend('Noisy Input d(n)', 'Cleaned Output e(n)', 'Location', 'best');
grid on;
xlim([0 1000]);

%% ------------------------------------------------------------------------
%  SNR Calculations
%  ------------------------------------------------------------------------
% Calculate Power of the desired signal
P_s = mean(s.^2);

% 1) Initial SNR of Corrupted Signal
noise_in = d - s;
P_noise_in = mean(noise_in.^2);
SNR_in = 10* log10(P_s / P_noise_in);

% 2) Final SNR of Cleaned Signal
noise_out = e(100:end) - s(100:end);
P_noise_out = mean(noise_out.^2);
P_s_converged = mean(s(100:end).^2);
SNR_out = 10 * log10(P_s_converged / P_noise_out);

SNR_improvement = SNR_out - SNR_in;

% Display SNR results
fprintf('\n--- Signal-to-Noise Ratio (SNR Results) ---\n');
fprintf('Initial SNR of Corrupted Signal: %8.2f dB\n', SNR_in);
fprintf('Final SNR of Cleaned Signal: %8.2f dB\n', SNR_out);
fprintf('SNR Improvement: %8.2f dB', SNR_improvement);
fprintf('\n-------------------------------------------\n');
