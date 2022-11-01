async function start() {

    const canvas = document.getElementById("my_canvas");
    //Inicialize the GL contex
    const gl = canvas.getContext("webgl2");
    if (gl === null) {
        alert("Unable to initialize WebGL. Your browser or machine may not support it.");
        return;
    }

    console.log("WebGL version: " + gl.getParameter(gl.VERSION));
    console.log("GLSL version: " + gl.getParameter(gl.SHADING_LANGUAGE_VERSION));
    console.log("Vendor: " + gl.getParameter(gl.VENDOR));

    let cameraSpeed = 0.05;

    const vsPoints = gl.createShader(gl.VERTEX_SHADER);
    const fsPoints = gl.createShader(gl.FRAGMENT_SHADER);
    const pointsProgram = gl.createProgram();

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

    // ========================================================= CAMERA POSES

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

    const vsCameraPose = gl.createShader(gl.VERTEX_SHADER);
    const fsCameraPose = gl.createShader(gl.FRAGMENT_SHADER);
    const cameraPoseProgram = gl.createProgram();

    gl.shaderSource(vsCameraPose, vsCameraPoseSource);
    gl.compileShader(vsCameraPose);
    if (!gl.getShaderParameter(vsCameraPose, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(vsCameraPose));
    }

    //compilation fs
    gl.shaderSource(fsCameraPose, fsCameraPoseSource);
    gl.compileShader(fsCameraPose);
    if (!gl.getShaderParameter(fsCameraPose, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(fsCameraPose));
    }

    if (!gl.getShaderParameter(fsCameraPose, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(fsCameraPose));
    }

    gl.attachShader(cameraPoseProgram, vsCameraPose);
    gl.attachShader(cameraPoseProgram, fsCameraPose);
    gl.linkProgram(cameraPoseProgram);

    if (!gl.getProgramParameter(cameraPoseProgram, gl.LINK_STATUS)) {
        console.log("ERROR")
        alert(gl.getProgramInfoLog(cameraPoseProgram));
    }

    const positionCameraPoseAttrib = gl.getAttribLocation(cameraPoseProgram, "position")
    gl.enableVertexAttribArray(positionCameraPoseAttrib);
    gl.vertexAttribPointer(positionCameraPoseAttrib, 3, gl.FLOAT, false, 3 * 4, 0);


    // ============================================== POINTS

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
               gl_PointSize = 1.0;
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


    //compilation vs
    gl.shaderSource(vsPoints, vsPointsSource);
    gl.compileShader(vsPoints);
    if (!gl.getShaderParameter(vsPoints, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(vsPoints));
    }

    //compilation fs
    gl.shaderSource(fsPoints, fsPointsSource);
    gl.compileShader(fsPoints);
    if (!gl.getShaderParameter(fsPoints, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(fsPoints));
    }

    if (!gl.getShaderParameter(fsPoints, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(fsPoints));
    }

    gl.attachShader(pointsProgram, vsPoints);
    gl.attachShader(pointsProgram, fsPoints);
    gl.linkProgram(pointsProgram);

    if (!gl.getProgramParameter(pointsProgram, gl.LINK_STATUS)) {
        console.log("ERROR")
        alert(gl.getProgramInfoLog(pointsProgram));
    }

    gl.useProgram(pointsProgram);

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

    const modelPointsLocation = gl.getUniformLocation(pointsProgram, "model")
    const projPointsLocation = gl.getUniformLocation(pointsProgram, "proj")
    const viewPointsLocation = gl.getUniformLocation(pointsProgram, "view")

    const modelCameraPoseLocation = gl.getUniformLocation(cameraPoseProgram, "model")
    const projCameraPoseLocation = gl.getUniformLocation(cameraPoseProgram, "proj")
    const viewCameraPoseLocation = gl.getUniformLocation(cameraPoseProgram, "view")

    let points = await fetchFile("http://localhost:5000/points.pts")
    let pointsNumber = await fetchFile("http://localhost:5000/points_in_frame.pts")
    let cameraPoses = await fetchFile("http://localhost:5000/camera_poses.pts")

    let framesNum = 1

    const pointsBuffer = gl.createBuffer()
    let pointsNum = loadPointsFromFrame(framesNum, points, pointsNumber)

    const positionPointsAttrib = gl.getAttribLocation(pointsProgram, "position")
    gl.enableVertexAttribArray(positionPointsAttrib);
    gl.vertexAttribPointer(positionPointsAttrib, 3, gl.FLOAT, false, 3 * 4, 0);

    const colorAttrib = gl.getAttribLocation(pointsProgram, "color")
    gl.enableVertexAttribArray(colorAttrib)
    gl.vertexAttribPointer(colorAttrib, 3, gl.FLOAT, false, 3 * 4, 3 * 4 )

    gl.uniformMatrix4fv(modelPointsLocation, false, model)
    gl.uniformMatrix4fv(projPointsLocation, false, proj)
    gl.uniformMatrix4fv(viewPointsLocation, false, view)

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

        // do not use yet
        //gl.useProgram(pointsProgram);

        gl.drawArrays(gl.POINTS, 0, pointsNum / 3);

        setCamera()

        mat4.lookAt(view, cameraPos, cameraFrontTmp, cameraUp);
        gl.uniformMatrix4fv(viewPointsLocation, false, view);
        gl.uniformMatrix4fv(modelPointsLocation, false, model)
        gl.uniformMatrix4fv(projPointsLocation, false, proj)

        setTimeout(() => { requestAnimationFrame(draw); }, 1000 / 120);
    }


    window.requestAnimationFrame(draw);

    // Add the event listeners for mousedown, mousemove, and mouseup
    window.addEventListener('mousedown', e => {
        x = e.offsetX;
        y = e.offsetY;
    });


    let cameraFrontTmp = glm.vec3(1, 1, 1);

    // Add the event listeners for keydown, keyup
    window.addEventListener('keyup', function (even) {
        pressedKey = -1
    }, false)
    window.addEventListener('keydown', function (event) {
        pressedKey = event.keyCode
    }, false);

    function setCameraMouse(e) {
        //Wyznaczyć zmianę pozycji myszy względem ostatniej klatki
        let xoffset = e.movementX;
        let yoffset = e.movementY;
        let sensitivity = 0.1;
        let cameraSpeed = 0.05 * elapsedTime;
        xoffset *= sensitivity;
        yoffset *= sensitivity;
        //Uaktualnić kąty
        yaw += xoffset * cameraSpeed;
        pitch -= yoffset * cameraSpeed;
        //Nałożyć ograniczenia co do ruchy kamery
        if (pitch > 89.0)
            pitch = 89.0;
        if (pitch < -89.0)
            pitch = -89.0;
        let front = glm.vec3(1, 1, 1);
        //Wyznaczenie wektora kierunku na podstawie kątów Eulera
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
                let cp = pointsNum
                pointsNum = loadPointsFromFrame(framesNum, points, pointsNumber)
                if (pointsNum == cp) {
                    framesNum--
                }
                console.log("points_num: ", pointsNum)
                break;
            case 69: // E 
                if (framesNum > 1) {
                    framesNum--
                    pointsNum = loadPointsFromFrame(framesNum, points, pointsNumber)
                    console.log("points_num: ", pointsNum)
                }
                break;
        }

        cameraFrontTmp.x = cameraPos.x + cameraFront.x;
        cameraFrontTmp.y = cameraPos.y + cameraFront.y;
        cameraFrontTmp.z = cameraPos.z + cameraFront.z;
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

    function loadPointsFromFrame(frame, points, pointsNums) {
        points = points.split("\n").flatMap(line => line.split(" ")).map(x => Number(x))
        pointsNums = pointsNums.split("\n").map(x => Number(x))

        let pointsToRead = 0
        let realSize = 0

        for (let i = 0; i < frame && i < pointsNums.length; i++) {
            pointsToRead += pointsNums[i]
        }

        tmpVec = new Array(pointsToRead)

        for (let i = 0; i < pointsToRead; i++) {
            for (let j = 0; j < 3; j++) {
                tmpVec[i*3+j] = points[i*3+j]
                realSize++
            }
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, pointsBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tmpVec), gl.STATIC_DRAW);

        return realSize
    }
}