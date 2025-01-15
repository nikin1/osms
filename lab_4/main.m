length_sequence = 31;

x = [0, 1, 1, 0, 0];
y = [1, 0, 0, 1, 1];

gold_sequence = create_gold_sequence(x, y, length_sequence);

fprintf('Последовательность Голда: ');
fprintf('%d ', gold_sequence);
fprintf('\n\n');

print_table(gold_sequence)

x = [0, 1, 1, 0, 1];
y = [0, 1, 1, 1, 0];

new_gold_sequence = create_gold_sequence(x, y, length_sequence);

fprintf('Новая последовательность Голда: ');
fprintf('%d ', new_gold_sequence);
fprintf('\n');

[corr, lags] = xcorr(gold_sequence, new_gold_sequence, 'normalized');
fprintf('Взаимная корреляция исходной и новой последовательности (0-й сдвиг): %.4f\n', corr(length_sequence));
fprintf('Взаимная корреляция исходной и новой последовательности (Максимальное знаечние): %.4f\n', max(corr));

function gold_sequence = create_gold_sequence(x, y, length_sequence)
    gold_sequence = zeros(1, length_sequence);

    for i = 1:length_sequence
        xor_shift_x = bitxor(x(3), x(5));
        xor_shift_y = bitxor(y(3), y(5));

        gold_sequence(i) = bitxor(x(end), y(end));

        x = [xor_shift_x, x(1:end-1)];
        y = [xor_shift_y, y(1:end-1)];
    end
end

function print_table(sequence)
    N = length(sequence);

    fprintf('Сдвиг | Последовательность                                            | Автокорреляция\n');
    fprintf('------|---------------------------------------------------------------|------------\n');
    
    [corr, lags] = autocorr(sequence, 'NumLags', N - 1);

    for i = 1:N
        shifted_sequence = circshift(sequence, i - 1);
       
        fprintf('%5d | ', lags(i));
        fprintf('%d ', shifted_sequence);
        fprintf('| %10.4f\n', corr(i));
    end

    fprintf('\n');
    figure;
    plot(0:N - 1, corr, '-o');
    xlabel('Задержка (lag)');
    ylabel('Автокорреляция');
    title('Функция автокорреляции в зависимости от задержки');
    xlim([0 N])
    grid on;
end
