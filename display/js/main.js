async function start() {
    const canvas = document.getElementById("my_canvas");
    const gl = canvas.getContext("webgl2");
    if (gl === null) {
        alert("Unable to initialize WebGL. Your browser or machine may not support it.");
        return;
    }

    console.log("WebGL version: " + gl.getParameter(gl.VERSION));
    console.log("GLSL version: " + gl.getParameter(gl.SHADING_LANGUAGE_VERSION));
    console.log("Vendor: " + gl.getParameter(gl.VENDOR));

    let cameraSpeed = 0.05;

    canvas.requestPointerLock = canvas.requestPointerLock ||
        canvas.mozRequestPointerLock;
    document.exitPointerLock = document.exitPointerLock ||
        document.mozExitPointerLock;
    canvas.onclick = function () {
        canvas.requestPointerLock();
    };
    // Hook pointer lock state change events for different browsers
    document.addEventListener('pointerlockchange', lockChangeAlert, false);
    document.addEventListener('mozpointerlockchange', lockChangeAlert, false);
    function lockChangeAlert() {
        if (document.pointerLockElement === canvas ||
            document.mozPointerLockElement === canvas) {
            console.log('The pointer lock status is now locked');
            document.addEventListener("mousemove", setCameraMouse, false);
        } else {
            console.log('The pointer lock status is now unlocked');
            document.removeEventListener("mousemove", setCameraMouse, false);
        }
    }
    document.addEventListener('wheel', event => {
        console.log(event)
        if (event.deltaY > 0) {
            cameraSpeed *= 0.5
        } else {
            cameraSpeed *= 1.5
        }
    })

    gl.enable(gl.DEPTH_TEST);

    let pressedKey
    let yaw = -90
    let pitch = 0


    const model = mat4.create();
    const rotation = 0;
    mat4.rotate(model, model, rotation, [0, 0, 1]);

    const view = mat4.create();
    mat4.lookAt(view, [0, 0, 3], [0, 0, -1], [0, 1, 0]);

    const proj = mat4.create()
    mat4.perspective(proj, 60 * Math.PI / 180, gl.canvas.clientWidth / gl.canvas.clientHeight, 0.1, 990000.0);

    let cameraPos = glm.vec3(0, 0, 3);
    let cameraFront = glm.vec3(0, 0, -1);
    let cameraUp = glm.vec3(0, 1, 0);

    let points = await fetchFile("http://localhost:5000/points.pts")
    let pointsNumber = await fetchFile("http://localhost:5000/points_in_frame.pts")
    let cameraPoses = await fetchFile("http://localhost:5000/camera_poses.pts")

    points = points.split("\n").flatMap(line => line.split(" ")).map(x => Number(x))
    pointsNumber = pointsNumber.split("\n").map(x => Number(x))
    cameraPoses = cameraPoses.split("\n").flatMap(line => line.split(" ")).map(x => Number(x))

    let pointsRenderer = new PointsRenderer(gl, points, pointsNumber, model, view, proj)
    let cameraPoseRenderer = new CameraPoseRenderer(gl, model, view, proj)

    let framesNum = 1

    let startTime = 0;
    let elapsedTime = 0;
    let licznik = 0;
    const fpsElem = document.getElementById("#fps");

    function draw() {
        elapsedTime = performance.now() - startTime;
        startTime = performance.now();
        licznik++;
        let fFps = 1000 / elapsedTime;

        if (licznik > fFps) {
            fpsElem.textContent = fFps.toFixed(1);
            licznik = 0;
        }

        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        pointsRenderer.draw(framesNum, model, view, proj)

        // to draw the same camera object in different poses always pass camera object 
        // and change location and rotation matrix for each camera pose drawing 

        setCamera()

        mat4.lookAt(view, cameraPos, cameraFrontTmp, cameraUp);
        gl.uniformMatrix4fv(pointsRenderer.viewLocation, false, view);
        gl.uniformMatrix4fv(pointsRenderer.modelLocation, false, model)
        gl.uniformMatrix4fv(pointsRenderer.projLocation, false, proj)

        setTimeout(() => { requestAnimationFrame(draw); }, 1000 / 120);
    }

    window.requestAnimationFrame(draw);

    window.addEventListener('mousedown', e => {
        x = e.offsetX;
        y = e.offsetY;
    });

    let cameraFrontTmp = glm.vec3(1, 1, 1);

    window.addEventListener('keyup', function (even) {
        pressedKey = -1
    }, false)
    window.addEventListener('keydown', function (event) {
        pressedKey = event.keyCode
    }, false);

    function setCameraMouse(e) {
        let xoffset = e.movementX;
        let yoffset = e.movementY;
        let sensitivity = 0.1;
        let cameraSpeed = 0.05 * elapsedTime;
        xoffset *= sensitivity;
        yoffset *= sensitivity;
        yaw += xoffset * cameraSpeed;
        pitch -= yoffset * cameraSpeed;
        if (pitch > 89.0)
            pitch = 89.0;
        if (pitch < -89.0)
            pitch = -89.0;
        let front = glm.vec3(1, 1, 1);
        front.x = Math.cos(glm.radians(yaw)) * Math.cos(glm.radians(pitch));
        front.y = Math.sin(glm.radians(pitch));
        front.z = Math.sin(glm.radians(yaw)) * Math.cos(glm.radians(pitch));
        cameraFront = glm.normalize(front);
    }

    function setCamera() {
        let cameraPos_tmp
        switch (pressedKey) {
            case 65: // Left
                cameraPos_tmp = glm.normalize(glm.cross(cameraFront, cameraUp));
                cameraPos.x -= cameraPos_tmp.x * cameraSpeed;
                cameraPos.y -= cameraPos_tmp.y * cameraSpeed;
                cameraPos.z -= cameraPos_tmp.z * cameraSpeed;
                break;
            case 87: // Up
                cameraPos.x += cameraSpeed * cameraFront.x;
                cameraPos.y += cameraSpeed * cameraFront.y;
                cameraPos.z += cameraSpeed * cameraFront.z;
                break;
            case 68: // Right
                cameraPos_tmp = glm.normalize(glm.cross(cameraFront, cameraUp));
                cameraPos.x += cameraPos_tmp.x * cameraSpeed;
                cameraPos.y += cameraPos_tmp.y * cameraSpeed;
                cameraPos.z += cameraPos_tmp.z * cameraSpeed;
                break;
            case 83: // Down
                cameraPos.x -= cameraSpeed * cameraFront.x;
                cameraPos.y -= cameraSpeed * cameraFront.y;
                cameraPos.z -= cameraSpeed * cameraFront.z;
                break;
            case 32: // Space
                cameraPos.y += cameraSpeed;
                break;
            case 90: // Z
                cameraPos.y -= cameraSpeed;
                break;
            case 82: // R
                framesNum++
                pointsRenderer.loadPointsFromFrame(framesNum)
                console.log("points_num: ", this.pointsNum)
                break;
            case 69: // E 
                if (framesNum > 1) {
                    framesNum--
                    pointsRenderer.loadPointsFromFrame(framesNum)
                    console.log("points_num: ", pointsRenderer.pointsNum)
                }
                break;
        }

        cameraFrontTmp.x = cameraPos.x + cameraFront.x;
        cameraFrontTmp.y = cameraPos.y + cameraFront.y;
        cameraFrontTmp.z = cameraPos.z + cameraFront.z;
    }
}

async function fetchFile(url) {
    return await fetch(url)
        .then(res => res.blob())
        .then(blob => blob.arrayBuffer())
        .then(arr => {
            var dec = new TextDecoder()
            return dec.decode(arr)
        })
        .catch(() => console.log("Failed to download file!"))
}

function loadPointsFromFrame(frame, cameraPoses) {
    tmpVec = new Array(pointsToRead)

    for (let i = 0; i < frames; i++) {
        for (let j = 0; j < 3; j++) {
            tmpVec[i * 3 + j] = points[i * 3 + j]
            realSize++
        }
    }

    return [15*frame, tmpVec]
}

function loadPointsFromFrame(frame, points, pointsNums) {
    let pointsToRead = 0
    let realSize = 0

    for (let i = 0; i < frame && i < pointsNums.length; i++) {
        pointsToRead += pointsNums[i]
    }

    tmpVec = new Array(pointsToRead)

    for (let i = 0; i < pointsToRead; i++) {
        for (let j = 0; j < 3; j++) {
            tmpVec[i * 3 + j] = points[i * 3 + j]
            realSize++
        }
    }

    return [realSize, tmpVec]
}

class PointsRenderer {
    constructor(gl, points, pointsNumber, model, view, proj) {
        this.gl = gl
        this.points = points
        this.pointsNumber = pointsNumber

        this.vsSource = vsPointsSource
        this.fsSource = fsPointsSource
        const vsPoints = gl.createShader(gl.VERTEX_SHADER);
        const fsPoints = gl.createShader(gl.FRAGMENT_SHADER);
        this.program = gl.createProgram();

        gl.shaderSource(vsPoints, this.vsSource);
        gl.compileShader(vsPoints);
        if (!gl.getShaderParameter(vsPoints, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(vsPoints));
        }

        gl.shaderSource(fsPoints, this.fsSource);
        gl.compileShader(fsPoints);
        if (!gl.getShaderParameter(fsPoints, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(fsPoints));
        }

        if (!gl.getShaderParameter(fsPoints, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(fsPoints));
        }

        gl.attachShader(this.program, vsPoints);
        gl.attachShader(this.program, fsPoints);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            console.log("ERROR")
            alert(gl.getProgramInfoLog(this.program));
        }

        gl.useProgram(this.program)

        this.modelLocation = gl.getUniformLocation(this.program, "model")
        this.projLocation = gl.getUniformLocation(this.program, "proj")
        this.viewLocation = gl.getUniformLocation(this.program, "view")

        this.buffer = gl.createBuffer()
        this.pointsNum = 0
        this.loadPointsFromFrame(1, points, pointsNumber) 
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(this.pointsVec), this.gl.STATIC_DRAW);

        this.positionAttrib = this.gl.getAttribLocation(this.program, "position")
        this.gl.enableVertexAttribArray(this.positionAttrib);
        this.gl.vertexAttribPointer(this.positionAttrib, 3, this.gl.FLOAT, false, 3 * 4, 0);

        // TODO: refactor
        this.colorAttrib = this.gl.getAttribLocation(this.program, "color")
        this.gl.enableVertexAttribArray(this.colorAttrib)
        this.gl.vertexAttribPointer(this.colorAttrib, 3, this.gl.FLOAT, false, 3 * 4, 3 * 4)

        gl.uniformMatrix4fv(this.modelLocation, false, model)
        gl.uniformMatrix4fv(this.projLocation, false, proj)
        gl.uniformMatrix4fv(this.viewLocation, false, view)
    }

    draw(frame, model, view, proj) {
        this.gl.useProgram(this.program)
        this.loadPointsFromFrame(frame, this.points, this.pointsNumber)

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(this.pointsVec), this.gl.STATIC_DRAW);
        
        this.gl.uniformMatrix4fv(this.modelLocation, false, model)
        this.gl.uniformMatrix4fv(this.projLocation, false, proj)
        this.gl.uniformMatrix4fv(this.viewLocation, false, view)

        this.gl.drawArrays(this.gl.POINTS, 0, this.pointsNum / 3);
    }

    loadPointsFromFrame(frame) {
        const [pointsNum, tmpVec] = loadPointsFromFrame(frame, this.points, this.pointsNumber)
        this.pointsNum = pointsNum
        this.pointsVec = tmpVec
    }
}

class CameraPoseRenderer {
    constructor(gl, cameraPoses, model, view, proj) {
        this.gl = gl
        this.vsSource = vsCameraPoseSource
        this.fsSource = fsCameraPoseSource
        const vsCameraPose = gl.createShader(gl.VERTEX_SHADER);
        const fsCameraPose = gl.createShader(gl.FRAGMENT_SHADER);
        this.program = gl.createProgram();

        gl.shaderSource(vsCameraPose, vsCameraPoseSource);
        gl.compileShader(vsCameraPose);
        if (!gl.getShaderParameter(vsCameraPose, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(vsCameraPose));
        }

        gl.shaderSource(fsCameraPose, fsCameraPoseSource);
        gl.compileShader(fsCameraPose);
        if (!gl.getShaderParameter(fsCameraPose, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(fsCameraPose));
        }

        if (!gl.getShaderParameter(fsCameraPose, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(fsCameraPose));
        }

        gl.attachShader(this.program, vsCameraPose);
        gl.attachShader(this.program, fsCameraPose);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            console.log("ERROR")
            alert(gl.getProgramInfoLog(this.program));
        }

        gl.useProgram(this.program)

        const positionAttrib = gl.getAttribLocation(this.program, "position")
        gl.enableVertexAttribArray(positionAttrib);
        gl.vertexAttribPointer(positionAttrib, 3, gl.FLOAT, false, 3 * 4, 0);

        this.modelLocation = gl.getUniformLocation(this.program, "model")
        this.projLocation = gl.getUniformLocation(this.program, "proj")
        this.viewLocation = gl.getUniformLocation(this.program, "view")
    }

    loadCameraPosesFromFrame(frame, model, view, proj) {
        
    }

    draw(frame) {
        this.gl.useProgram(this.program)
        this.loadPointsFromFrame(frame, this.points, this.pointsNumber)

        this.gl.uniformMatrix4fv(this.modelLocation, false, this.model)
        this.gl.uniformMatrix4fv(this.projLocation, false, this.proj)
        this.gl.uniformMatrix4fv(this.viewLocation, false, this.view)

        this.gl.drawArrays(this.gl.POINTS, 0, this.pointsNum / 3);
    }
}

const vsCameraPoseSource =
    `#version 300 es
    precision highp float;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 proj;

    in vec3 position;
    in vec4 firstRow;
    in vec4 secondRow;
    in vec4 thirdRow;

    void main() {
        vec4 vec = vec4(0.0, 0.0, 0.0, 1.0);
        mat4 rot = mat4(firstRow, secondRow, thirdRow, vec);
        gl_Position = rot * proj * view * model * vec4(position, 1.0);
    }
    `

const fsCameraPoseSource =
    `#version 300 es
    precision highp float;

    out vec4 Color;

    void main() {
        Color = vec4(0.0, 0.1, 0.0, 1.0);
    }
    `

const vsPointsSource =
    `#version 300 es
			precision highp float;

			uniform mat4 model;
			uniform mat4 view;
			uniform mat4 proj;

			in vec3 position;
            in vec3 color;

            out vec3 pointColor;

			void main(void)
			{
			   gl_Position = proj * view * model * vec4(position, 1.0);
               gl_PointSize = 2.0;
               pointColor = color;
			}
			`;

const fsPointsSource =
    `#version 300 es
		    precision highp float;

            in vec3 pointColor;
		    out vec4 Color;

		    void main(void)
			{
                Color = vec4(pointColor, 1.0);
	   		}
			`;
