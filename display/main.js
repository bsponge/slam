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

    const vs = gl.createShader(gl.VERTEX_SHADER);
    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    const program = gl.createProgram();

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
            document.addEventListener("mousemove", ustaw_kamere_mysz, false);
        } else {
            console.log('The pointer lock status is now unlocked');
            document.removeEventListener("mousemove", ustaw_kamere_mysz, false);
        }
    }

    gl.enable(gl.DEPTH_TEST);

    let pressedKey
    let yaw = -90; //obrót względem osi X
    let pitch = 0; //obrót względem osi Y

    const vsSource =
        `#version 300 es
			precision highp float;

			uniform mat4 model;
			uniform mat4 view;
			uniform mat4 proj;

			in vec3 position;

			void main(void)
			{
			   gl_Position = proj * view * model * vec4(position, 1.0);
               gl_PointSize = 3.0;
			}
			`;

    const fsSource =
        `#version 300 es
		    precision highp float;

		    out vec4 Color;
            uniform vec3 color;

		    void main(void)
			{
                Color = vec4(color, 1.0);
	   		}
			`;


    //compilation vs
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(vs));
    }

    //compilation fs
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(fs));
    }

    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(fs));
    }

    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.log("ERROR")
        alert(gl.getProgramInfoLog(program));
    }

    gl.useProgram(program);

    const model = mat4.create();
    const rotation = -25 * Math.PI / 180;
    mat4.rotate(model, model, rotation, [0, 0, 1]);

    const view = mat4.create();
    mat4.lookAt(view, [0, 0, 3], [0, 0, -1], [0, 1, 0]);

    const proj = mat4.create()
    mat4.perspective(proj, 60 * Math.PI / 180, gl.canvas.clientWidth / gl.canvas.clientHeight, 01, 100.0);

    let cameraPos = glm.vec3(0, 0, 3);
    let cameraFront = glm.vec3(0, 0, -1);
    let cameraUp = glm.vec3(0, 1, 0);
    let obrot = 0.0;

    const modelLocation = gl.getUniformLocation(program, "model")
    const projLocation = gl.getUniformLocation(program, "proj")
    const viewLocation = gl.getUniformLocation(program, "view")
    const colorLocation = gl.getUniformLocation(program, "color")


    const buffer = gl.createBuffer();
    let points = await fetchFile("http://localhost:5000/points")
    let pointsNumber = await fetchFile("http://localhost:5000/points_in_frame")

    let framesNum = 2;
    let pointsNum = loadPointsFromFrame(framesNum, points, pointsNumber)
    //kostka()

    const positionAttrib = gl.getAttribLocation(program, "position")
    gl.enableVertexAttribArray(positionAttrib);
    gl.vertexAttribPointer(positionAttrib, 3, gl.FLOAT, false, 6 * 4, 0);


    gl.uniformMatrix4fv(modelLocation, false, model)
    gl.uniformMatrix4fv(projLocation, false, proj)
    gl.uniformMatrix4fv(viewLocation, false, view)
    gl.uniform3f(colorLocation, 1.0, 1.0, 1.0)

    let startTime = 0;
    let elapsedTime = 0;
    let licznik = 0;
    let FPS = 90;
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


        gl.drawArrays(gl.POINTS, 0, pointsNum/3);

        ustaw_kamere()

        mat4.lookAt(view, cameraPos, cameraFrontTmp, cameraUp);
        gl.uniformMatrix4fv(viewLocation, false, view);

        setTimeout(() => { requestAnimationFrame(draw); }, 1000 / 120);
    }


    window.requestAnimationFrame(draw);

    // Add the event listeners for mousedown, mousemove, and mouseup
    window.addEventListener('mousedown', e => {
        x = e.offsetX;
        y = e.offsetY;
    });

    let cameraSpeed = 0.05;
    let cameraFrontTmp = glm.vec3(1, 1, 1);

    // Add the event listeners for keydown, keyup
    window.addEventListener('keyup', function (even) {
        pressedKey = -1
    }, false)
    window.addEventListener('keydown', function (event) {
        pressedKey = event.keyCode
    }, false);

    function ustaw_kamere_mysz(e) {
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

    function ustaw_kamere() {
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

    function kostka() {

        let punkty_ = 36;

        var vertices = [
            -0.5, -0.5, -0.5, 0.0, 0.0, 0.0,
            0.5, -0.5, -0.5, 0.0, 0.0, 1.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 1.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 1.0,
            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0,
            -0.5, -0.5, -0.5, 0.0, 0.0, 0.0,

            -0.5, -0.5, 0.5, 0.0, 0.0, 0.0,
            0.5, -0.5, 0.5, 1.0, 0.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 1.0, 0.0,
            -0.5, -0.5, 0.5, 0.0, 0.0, 0.0,

            -0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
            -0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
            -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
            -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
            -0.5, -0.5, 0.5, 0.0, 0.0, 0.0,
            -0.5, 0.5, 0.5, 1.0, 0.0, 1.0,

            0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
            0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
            0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
            0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
            0.5, -0.5, 0.5, 0.0, 0.0, 0.0,
            0.5, 0.5, 0.5, 1.0, 0.0, 1.0,

            -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
            0.5, -0.5, -0.5, 1.0, 1.0, 1.0,
            0.5, -0.5, 0.5, 1.0, 0.0, 1.0,
            0.5, -0.5, 0.5, 1.0, 0.0, 1.0,
            -0.5, -0.5, 0.5, 0.0, 0.0, 0.0,
            -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,

            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0,
            0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 0.0, 0.0,
            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0
        ];

        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        n_draw = punkty_;
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

        let vector = []
        let pointsToRead = 0
        let realSize = 0

        for (let i = 0; i < frame && i < pointsNums.length; i++) {
            pointsToRead += pointsNums[i]
        }

        tmpVec = new Array(pointsToRead)
        let maxX = Number.MIN_VALUE
        let maxY = maxX
        let maxZ = maxX

        for (let i = 0; i < pointsToRead; i++) {
            for (let j = 0; j < 3; j++) {
                tmpVec[i*3+j] = points[i*3+j]
                realSize++
            }

            if (Math.abs(tmpVec[i]) > maxX && i < realSize) {
                maxX = Math.abs(tmpVec[i])
            }
            if (Math.abs(tmpVec[i+1]) > maxY && i < realSize+1) {
                maxY = Math.abs(tmpVec[i+1])
            }
            if (Math.abs(tmpVec[i+2]) > maxZ && i < realSize+2) {
                maxZ = Math.abs(tmpVec[i+2])
            }
        }

        for (let i = 0; i < realSize; i+=3) {
            tmpVec[i] /= 1000
            tmpVec[i+1] /= 1000
            tmpVec[i+2] /= 1000
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tmpVec), gl.STATIC_DRAW);

        return realSize
    }
}