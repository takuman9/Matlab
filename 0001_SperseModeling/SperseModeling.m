%% 設定周り

% シードの設定
rng = 0;

% 特徴量とサンプルサイズ
n_features      = 1000; % 特徴量
n_samples       = 100;  % サンプル数
n_nonzero_coefs = 20;   % 有効な係数
n_iter          = 50;   % 繰り返し回数

% 正規化パラメータ
alpha = 0.75;

%% 真の重み生成
idx    = randi([0, n_features], 1,n_nonzero_coefs); % idx：0-1000までのうちランダムに20個決める)
w      = zeros(n_features,1);                       % w  ：1000の特徴量分の係数の箱をつくる
w(idx) = randn(n_nonzero_coefs,1);                  % w  ：ランダムな値を20個の係数に値を入れる


%% 入力データと観測情報
X = randn(n_samples,n_features); % X：100ｘ1000
y = X*w + randn(n_samples,1);    % y：[100ｘ1000]*[1000ｘ1]＋δ*[100ｘ1]　=　[100ｘ1]


%% グラフの描画
ax1 = subplot(2,2,1);
ax2 = subplot(2,2,2);
stem(ax1,w);
plot(ax2,y);
title(ax1,'coeficients');
title(ax2,'value');

%% LASSO実装（ADMM）
[pred, w_]= fr_lasso_admm(X,y,alpha,n_iter);


%% グラフの描画
ax2 = subplot(2,2,4);
ax3 = subplot(2,2,3);
ax4 = subplot(2,2,4);
stem(ax3,[w,w_]);
plot(ax2,y, 'Color','blue');hold on;
plot(ax2,pred,'Color','red' );hold off;
title(ax1,'coeficients');
title(ax2,'value');
title(ax3,'sperse coef');
title(ax4,'sperse value');
