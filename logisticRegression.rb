include Math

def hypothesis_function(theta, x, n)
  theta_transpose_x = 0
  0.upto n do |i|
    theta_transpose_x += theta[i] * x[i]
  end

  1.0 / (1 + exp(-theta_transpose_x))
end

def cost_function(theta, x, y, m, n)
  summation = 0.0
  0.upto m-1 do |i|
    summation += y[i] * log(hypothesis_function(theta, x[i], n)) +
        (1 - y[i]) *
            log(1 - hypothesis_function(theta, x[i], n))
  end
  -summation / m
end

def gradientdescent(theta, x, y, m, n, alpha, iterations)
  0.upto iterations-1 do |i|
    thetatemp = theta.clone
    0.upto n do |j|
      summation = 0.0
      0.upto m-1 do |k|
        summation += (hypothesis_function(theta, x[k], n) - y[k]) *
            x[k][j]
      end
      thetatemp[j] = thetatemp[j] - alpha * summation / m
    end
    theta = thetatemp.clone
  end
  theta
end

def scalefeatures(data, m, n)
  mean = [0]
  1.upto n do |j|
    sum = 0.0
    0.upto m-1 do |i|
      sum += data[i][j]
    end
    mean << sum / m
  end
  stddeviation = [0]
  1.upto n do |j|
    temp = 0.0
    0.upto m-1 do |i|
      temp += (data[i][j] - mean[j]) ** 2
    end
    stddeviation << sqrt(temp / m)
  end
  1.upto n do |j|
    0.upto m-1 do |i|
      data[i][j] = (data[i][j] - mean[j]) / stddeviation[j]
    end
  end
  [data, mean, stddeviation]
end

def train_logistic_model
  x, y = [], []

  # 加载第一组训练样本
  file = File.open('train_class1.txt', 'rb')
  features = file.read.split(',')
  i = 0
  while i < features.length
    a = i + 1
    b = i + 101
    x << features[a..b].map{  |item| Float(item) }
    y << 0
    i = i + 102
  end
  file.close

  # 加载第二组训练样本
  file = File.open('train_class2.txt', 'rb')
  features = file.read.split(',')
  i = 0
  while i < features.length
    a = i + 1
    b = i + 101
    x << features[a..b].map{  |item| Float(item) }
    y << 1
    i = i + 102
  end
  file.close

  puts('Starting Training Logistic Regression Model')

  m = x.length      # 样本数
  n = x[0].length   # features
  # x0
  x.each {|i| i.unshift(1)}

  initialtheta = [0.0] * (n + 1)
  learningrate = 0.01
  iterations   = 100
  scale = scalefeatures(x, m, n)
  x = scale[0]

  finaltheta = gradientdescent(initialtheta,
                                        x, y, m, n,
                                        learningrate, iterations)

  puts "Initial cost: #{cost_function(initialtheta, x, y, m, n)}"
  puts "Final cost: #{cost_function(finaltheta, x, y, m, n)}"

  file = File.open('finaltheta.txt', 'w+b')
  file.write(finaltheta.join(','))
  file.close
  file = File.open('scale.txt', 'w+b')
  file.write((scale[1] + scale[2]).join(','))
  file.close
end
