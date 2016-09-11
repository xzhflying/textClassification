require 'stopwords'

# 把指定目录下的文件去除一些统一的header
def file_preprocess(dir)
  # 停止词的filter
  filter = Stopwords::Snowball::Filter.new "en"

  Dir.foreach(dir) do |item|
    next if item == '.' or item == '..' or item == '.DS_Store'
    path = dir + String(item)
    file = File.open(path, 'rb')
    content = file.read.downcase.split(/\W+/)

    0.upto(content.length - 1){ |i|
      if content[i] == "lines"
        content = content.drop(i + 2)
        break
      end
    }
    file.close

    filter.filter(content)
    file = File.open(path, 'w+b')
    file.write(content.join(' '))
    file.close
  end
end

# 对于word vector的得出。把所有训练样本文件合成为一个文件，并做一些简单的预处理
def word_vectors_preprocess
  dir = 'train/word_vectors/'

  file_preprocess(dir)

  Dir.foreach(dir) do |item|
    next if item == '.' or item == '..' or item == '.DS_Store'
    path = dir + String(item)
    file = File.open(path, 'rb')
    content = file.read
    file.close
    file = File.open("word_vectors_tempfile.txt", 'a')
    file.write(content + ' ')
    file.close
  end
end

# 对于logistic regression的预处理。把每一个训练文件里能够找到的word vector求和取平均，作为一个文件的feature
def logistic_regression_preprocess
  dir1 = 'train/logistic/class1/'
  dir2 = 'train/logistic/class2/'
  file_preprocess(dir1)
  file_preprocess(dir2)

  # 加载word vector
  word_vector_list = Hash.new
  file = File.open('word_vectors.txt', 'rb')
  vectors = file.read.split(',')
  i = 0
  while i < vectors.length
    a = i + 1
    b = i + 101
    word_vector_list[vectors[i]] = vectors[a..b]
    i = i + 102
  end
  file.close

  file_to_features(dir1, word_vector_list, 'train_class1.txt')
  file_to_features(dir2, word_vector_list, 'train_class2.txt')
end

# 利用词向量表将对应文件夹下文件转化为一张file name feature表
def file_to_features(dir, word_vector_list, output_file)
  Dir.foreach(dir) do |item|
    next if item == '.' or item == '..' or item == '.DS_Store'
    path = dir + String(item)
    file = File.open(path, 'rb')
    content = file.read.split(/\W+/)
    features = Array.new(101, 0.0)
    count = 0
    for i in content
      next if word_vector_list[i].nil?
      0.upto(word_vector_list[i].length - 1){ |j|
        features[j] += Float(word_vector_list[i][j])
      }
      count += 1
    end
    file.close
    next if count == 0
    0.upto(features.length - 1){  |i| features[i] = features[i] / Float(count)}

    file = File.open(output_file, 'a')
    file.write(item + ',' + features.join(',') + ',')
    file.close
  end
end

def test_preprocess
  dir = 'test/'
  file_preprocess(dir)

  # 加载word vector
  word_vector_list = Hash.new
  file = File.open('word_vectors.txt', 'rb')
  vectors = file.read.split(',')
  i = 0
  while i < vectors.length
    a = i + 1
    b = i + 101
    word_vector_list[vectors[i]] = vectors[a..b]
    i = i + 102
  end
  file.close

  file_to_features(dir, word_vector_list, 'test_features.txt')
end
