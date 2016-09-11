load 'preprocess.rb'
load 'logisticRegression.rb'
load 'word2vec.rb'

def train_related_models
  word_vectors_preprocess
  train_vocabulary_list
  logistic_regression_preprocess
  train_logistic_model
  puts 'Finish Training Models'
end

def test_result
  test_preprocess
  file = File.open('finaltheta.txt', 'rb')
  finaltheta = file.read.split(',').map!{  |item| Float(item) }
  file.close

  # 用来scale
  scale = Array.new
  file = File.open('scale.txt', 'rb')
  s = file.read.split(',')
  half = s.length / 2
  scale[0] = s[0..(half - 1)].map{  |item| Float(item) }
  scale[1] = s[half..(s.length - 1)].map{  |item| Float(item) }
  file.close

  file = File.open('test_features.txt', 'rb')
  content = file.read.split(',')
  file.close

  result_hash = Hash.new
  i = 0
  while i < content.length
    a = i + 1
    b = i + 101
    x = content[a..b].map{  |item| Float(item) }
    x.unshift(1)
    n = x.length - 1
    1.upto(n - 1) { |i| x[i] = (x[i] - scale[0][i]) / scale[1][i] }
    result = hypothesis_function(finaltheta, x, n)
    if result <= 0.5
      result_hash[content[i]] = 0
    else
      result_hash[content[i]] = 1
    end
    i = i + 102
  end

  class1, class2 = [], []
  result_hash.each{ |key, value|
    class1 << key if value == 0
    class2 << key if value == 1
  }

  correct = 0
  count = 0
  Dir.foreach('answer/class1/') do |item|
    next if item == '.' or item == '..' or item == '.DS_Store'
    correct += 1 if class1.include?(String(item))
    count += 1
  end
  correct_rate = Float(correct) / Float(count)
  puts "There are #{count} file in class1. Correct number is #{correct}. Correct rate is #{correct_rate}"

  correct = 0
  count = 0
  Dir.foreach('answer/class2/') do |item|
    next if item == '.' or item == '..' or item == '.DS_Store'
    correct += 1 if class2.include?(String(item))
    count += 1
  end
  correct_rate = Float(correct) / Float(count)
  puts "There are #{count} file in class2. Correct number is #{correct}. Correct rate is #{correct_rate}"
end

# train_related_models
test_result
