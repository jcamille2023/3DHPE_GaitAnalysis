
function gait_measures = get_gait_measures(filename)
    
    % Read data
    data0 = readtable(filename);
    
    % Extract X, Y, Z coordinates for left side 
    l_hip_data0 = data0(data0.joint == 23, :);
    l_foot_data0 = data0(data0.joint == 29, :);
    
    % Calculate X-direction distance for each dataset
    l_x_distance0 = abs(l_hip_data0.x - l_foot_data0.x); % x column
    
    % Shift and combine timestamps
    l_timestamp0 = l_foot_data0.timestamp; 
    
    % To make sure there are no duplicates and that time only increases
    l_timestamp0 = sort(l_timestamp0); 
    
    % Define filter parameters
    fs = 1 / mean(diff(l_timestamp0)); % Sampling frequency
    cutoff = 1; % Cutoff frequency (1 Hz) 1 - 2 Hz best for walking -- justify?
    order = 6; % Butterworth lowpass filter order (higher = smoother)
    
    % Design Butterworth lowpass filter
    [b, a] = butter(order, cutoff / (fs / 2), 'low');
    
    % Apply the filter to the distance data
    l_smoothed_x_distance = filtfilt(b, a, l_x_distance0);
    
    % Detect peaks (heel strike) and minima (toe-off) in smoothed data
    [peaks_l, peak_locs_l] = findpeaks(l_smoothed_x_distance, l_timestamp0);
    [minima_l, min_locs_l] = findpeaks(-l_smoothed_x_distance, l_timestamp0);
    
    
    figure;
    plot(l_timestamp0, l_smoothed_x_distance, '-');
    xlabel('Time (s)');
    ylabel('Smoothed Distance (X direction)');
    title('Smoothed Distance between Hip (Joint 23) and Foot (Joint 29)');
    grid on;
    
    % Apply thresholds for peaks and minima
    peak_threshold_l = max(l_smoothed_x_distance) * 0.35; % 35% threshold for peak
    valid_peaks_idx_l = peaks_l >= peak_threshold_l;
    peaks_l = peaks_l(valid_peaks_idx_l);
    peak_locs_l = peak_locs_l(valid_peaks_idx_l);
    
    min_threshold_l = min(l_smoothed_x_distance) * 0.18; % 18% threshold for minima
    valid_minima_idx_l = minima_l <= -min_threshold_l;
    minima_l = minima_l(valid_minima_idx_l);
    min_locs_l = min_locs_l(valid_minima_idx_l);
    
    hold on;
    plot(peak_locs_l, peaks_l, 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 8, 'DisplayName', 'Heel Strike');
    plot(min_locs_l, -minima_l, 'x', 'MarkerEdgeColor', 'b', 'MarkerSize', 8, 'DisplayName', 'Toe-Off');
    hold off;
    % Calculate timings and durations
    stance_times_l = arrayfun(@(i) abs(min_locs_l(i) - peak_locs_l(i)), 1:length(peak_locs_l)-1);
    swing_times_l = arrayfun(@(i) abs(peak_locs_l(i) - min_locs_l(i+1)), 1:length(min_locs_l)-1);
    
    % Extract X, Y, Z coordinates for right side 
    r_hip_data0 = data0(data0.joint == 24, :);
    r_foot_data0 = data0(data0.joint == 30, :);
    
    % Calculate X-direction distance for each dataset
    r_x_distance0 = abs(r_hip_data0.x - r_foot_data0.x); % x column
    r_timestamp0 = r_foot_data0.timestamp; 
    
    % Apply the filter to the distance data
    r_smoothed_x_distance = filtfilt(b, a, r_x_distance0);
    
    % Detect peaks (heel strike) and minima (toe-off) in smoothed data
    [peaks_r, peak_locs_r] = findpeaks(r_smoothed_x_distance, r_timestamp0);
    [minima_r, min_locs_r] = findpeaks(-r_smoothed_x_distance, r_timestamp0);
    
    figure;
    plot(r_timestamp0, r_smoothed_x_distance, '-');
    xlabel('Time (s)');
    ylabel('Smoothed Distance (X direction)');
    title('Smoothed Distance between Hip (Joint 24) and Foot (Joint 30)');
    grid on;
    
    % Apply thresholds for peaks and minima
    peak_threshold_r = max(r_smoothed_x_distance) * 0.35; % 35% threshold for peak
    valid_peaks_idx_r = peaks_r >= peak_threshold_r;
    peaks_r = peaks_r(valid_peaks_idx_r);
    peak_locs_r = peak_locs_r(valid_peaks_idx_r);
    
    min_threshold_r = min(r_smoothed_x_distance) * 0.18; % 18% threshold for minima
    valid_minima_idx_r = minima_r <= -min_threshold_r;
    minima_r = minima_r(valid_minima_idx_r);
    min_locs_r = min_locs_r(valid_minima_idx_r);
    
    
    hold on;
    plot(peak_locs_r, peaks_r, 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 8, 'DisplayName', 'Heel Strike');
    plot(min_locs_r, -minima_r, 'x', 'MarkerEdgeColor', 'b', 'MarkerSize', 8, 'DisplayName', 'Toe-Off');
    hold off;
    
    % Calculate timings and durations
    stance_times_r = arrayfun(@(i) abs(min_locs_r(i) - peak_locs_r(i)), 1:length(peak_locs_r)-1);
    swing_times_r = arrayfun(@(i) abs(peak_locs_r(i+1) - min_locs_r(i)), 1:length(min_locs_r)-1);
    
    % Calculate step time and double support time
    step_times = arrayfun(@(i) abs(peak_locs_r(i) - peak_locs_l(i)), 1:min(length(peak_locs_r), length(peak_locs_l)));
    double_support_times = arrayfun(@(i) abs(min_locs_l(i) - peak_locs_r(i)), 1:min(length(min_locs_l), length(peak_locs_r)));
    
    % Calculate step length and stride length
    step_lengths = []
    for i = 1:min(height(l_peak_points),height(r_peak_points))
        step_lengths(i) = sqrt((l_peak_points{i,3}-r_peak_points{i,3})^2 +(l_peak_points{i,4}-r_peak_points{i,4})^2 + (l_peak_points{i,5}-r_peak_points{i,5})^2)
    end
    l_stride_lengths = []
    for i = 2:height(l_peak_points)
        l_stride_lengths(i) = sqrt((l_peak_points{i-1,3}-l_peak_points{i,3})^2+(l_peak_points{i-1,4}-l_peak_points{i,4})^2+(l_peak_points{i-1,5}-l_peak_points{i,5})^2)
    end
    r_stride_lengths = []
    for i = 2:height(r_peak_points)
        r_stride_lengths(i) = sqrt((r_peak_points{i-1,3}-r_peak_points{i,3})^2+(r_peak_points{i-1,4}-r_peak_points{i,4})^2+(r_peak_points{i-1,5}-r_peak_points{i,5})^2)
    end
    
    % Calculate cadence (steps per second)
    cadence = length(peak_locs_l) / (l_timestamp0(end) - l_timestamp0(1));
    
    % Display results
    fprintf('Average Stance Time (Left): %.2f s\n', mean(stance_times_l));
    fprintf('Average Swing Time (Left): %.2f s\n', mean(swing_times_l));
    fprintf('Average Stance Time (Right): %.2f s\n', mean(stance_times_r));
    fprintf('Average Swing Time (Right): %.2f s\n', mean(swing_times_r));
    fprintf('Average Step Time: %.2f s\n', mean(step_times));
    fprintf('Average Double Support Time: %.2f s\n', mean(double_support_times));
    fprintf('Average Step Length: %.2f m\n', mean(step_lengths));
    fprintf('Average Stride Length (l): %.2f m\n', mean(l_stride_lengths));
    fprintf('Average Stride Length (r): %.2f m\n', mean(r_stride_lengths));
    fprintf('Cadence: %.2f steps/s\n', cadence);

    gait_measures = dictionary(["stance_time_l","swing_time_l","stance_time_r", ...
                                "swing_time_r","step_time","double_support_time", ...
                                "step_length","stride_length","cadence"], ...
                                [mean(stance_times_l),mean(swing_times_l),mean(stance_times_r), ...
                                mean(swing_times_r),mean(step_times),mean(double_support_times), ...
                                mean(step_lengths),mean(l_stride_lengths,mean(r_stride_lengths))])
end