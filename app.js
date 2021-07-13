var image = require('get-image-data')
 
image('./Untitled-1.jpg', function (err, info) {
  var data = info.data
  var height = info.height
  var width = info.width
 console.log(info);
  for (var i = 0, l = data.length; i < l; i += 4) {
    var red = data[i]
    var green = data[i + 1]
    var blue = data[i + 2]
    var alpha = data[i + 3]
  }
  // console.log(red);
  // console.log(green);
  // console.log(blue);
  // console.log(alpha);
})