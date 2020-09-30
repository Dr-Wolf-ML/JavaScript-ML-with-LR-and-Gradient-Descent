const { sigmoid, softmax } = require('@tensorflow/tfjs');
// require('@tensorflow/tfjs-node');  // threw error in OSX.14.beta.6
const tf = require('@tensorflow/tfjs');

// LogisticRegression expects features to be [[], [], ...]
class LogisticRegression {
  constructor(features, labels, options) {
    // expected to be passed as tensorflow Tensors:
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];  // 'cost...' is the scientific term used in literature for Cross Entropy
    this.bHistory = [];

    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 100,
      batchSize: 50,
      classificationFunction: 'softmax'  // can be either 'softmax' or 'sigmoid'
    }, options);

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  applyClassificationFunction(tensor) {
    return this.options.classificationFunction === 'softmax' ? tensor.softmax() : tensor.sigmoid();
  }

  gradientDescent(features, labels) {
    let currentGuesses = features
      .matMul(this.weights);
    
    currentGuesses = this.applyClassificationFunction(currentGuesses);

    const differences = currentGuesses.sub(labels);
    
    const slopes = features  // slopes is sometimes called 'gradients'
      .transpose()
      .matMul(differences)
      .div(features.shape[0])
      // .mul(2) is required in the original equation, but doesn't add any real value here !!

    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
      );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const {batchSize} = this.options;
      
        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [batchSize, -1]
          );
          const labelSlice = this.labels.slice(
            [startIndex, 0],
            [batchSize, -1]
          );

          this.bHistory.push(this.weights.arraySync()[0][0]);
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }

      this.recordCost();
      this.updateLearningRate();
    }
  }
  
  predict(observations) {
    const term = this.processFeatures(observations)
      .matMul(this.weights)

    return this.applyClassificationFunction(term)
        .argMax(1);
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels)
      .argMax(1);

    const incorrect = predictions
      .notEqual(testLabels)
      .sum()
      .arraySync();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features  = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const {mean, variance} = tf.moments(features, 0);

    // debugging variance
    const filler = variance.cast('bool').logicalNot().cast('float32');

    this.mean = mean;
    this.variance = variance.add(filler);

    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const cost = tf.tidy(() => {

      let guesses = this.features
        .matMul(this.weights);

      guesses = this.applyClassificationFunction(guesses);

      const termOne = this.labels
        .transpose()
        .matMul(
          guesses
            .add(1e-7)  // bug fix to prevent a log(0) to be taken in the next step
            .log()
          );

      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(
          guesses
            .mul(-1)
            .add(1)
            .add(1e-7)  // bug fix to prevent a log(0) to be taken in the next step
            .log()
          );

      return [termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .bufferSync()
        .values[0]];
    });

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    } 
    
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
