// var express = require('express')
// var webcam = express()
// var server = require('http').Server(webcam)
// var socketIO = require('socket.io')
// var fs = require('fs')

// var io = socketIO(server)





// function decodeBase64Image(dataString) {
//     var matches = dataString.match(/^data:([A-Za-z-+\/]+);base64,(.+)$/),
//         response = {};

//     if (matches.length !== 3) {
//         return new Error('Invalid input string');
//     }

//     response.type = matches[1];
//     response.data = new Buffer(matches[2], 'base64');

//     return response;
// }



// server.listen(3000, function () {
//     console.log('listening to requist on port 3000')
// })

// webcam.use(express.static('public'))

// io.on('connection', function (socket) {
//     console.log('made socket connection')
//     socket.on('responce', (data) => {
//         console.log('link is given below')

//         // var imageBuffer = decodeBase64Image(data.data);
//         let image= data.data

//         var LKJH = image.replace(/^data:image\/\w+;base64,/, '');

//         // console.log(imageBuffer)

//         fs.writeFile('fileName.JPG', LKJH, { encoding: 'base64' }, function (err) {
//             //Finished
//             console.log('file written')
//         });

//         // fs.writeFile('test.jpeg', imageBuffer.data, function (err) { console.log('file written') });
//         // console.log(data)
//     })


//     socket.emit('server_se_le', { data: 'abc123' })
// })