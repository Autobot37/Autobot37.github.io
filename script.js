const canvas = document.getElementById('canvas1');
const ctx = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

ctx.fillStyle = 'white';
ctx.strokeStyle = 'white';
ctx.lineWidth = 1;

class Particle {
    constructor(effect) {
        this.effect = effect;
        this.reset();
    }

    reset() {
        this.x = Math.floor(Math.random() * this.effect.width);
        this.y = Math.floor(Math.random() * this.effect.height);
        this.speedX = 0;
        this.speedY = 0;
        this.speedModifier = Math.floor(Math.random() * 5 + 1);
        this.history = [{ x: this.x, y: this.y }];
        this.maxLength = Math.floor(Math.random() * 100 + 10);
        this.angle = 0;
        this.timer = this.maxLength * 2;
        this.colors = ['#3d03fc', '', '#032cfc', '#036bfc', '#6b03fc'];
        this.color = this.colors[Math.floor(Math.random() * this.colors.length)];
    }

    draw(context) {
        context.beginPath();
        context.moveTo(this.history[0].x, this.history[0].y);
        for (let i = 0; i < this.history.length; i++) {
            context.lineTo(this.history[i].x, this.history[i].y);
        }
        context.strokeStyle = this.color;
        context.stroke();
    }

    update() {
        this.timer--;
        if (this.timer >= 1) {
            const x = Math.floor(this.x / this.effect.cellSize);
            const y = Math.floor(this.y / this.effect.cellSize);
            const index = y * this.effect.cols + x;
            const angle = this.effect.flowField[index];

            this.speedX = Math.cos(angle);
            this.speedY = Math.sin(angle);
            this.x += this.speedX * this.speedModifier;
            this.y += this.speedY * this.speedModifier;

            this.history.push({ x: this.x, y: this.y });
            if (this.history.length > this.maxLength) {
                this.history.shift();
            }
        } else if (this.history.length > 1) {
            this.history.shift();
        } else {
            this.reset();
        }
    }
}

class Effect {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.particles = [];
        this.numberOfParticles = 2000;
        this.cellSize = 20;
        this.rows = Math.floor(this.height / this.cellSize);
        this.cols = Math.floor(this.width / this.cellSize);
        this.flowField = [];
        this.curve = 0.7;
        this.zoom = 0.09;
        this.debug = true;
        this.init();

        window.addEventListener('keydown', (e) => {
            console.log(e);
        });
    }

    init() {
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const angle = Math.cos(x * this.zoom) + Math.sin(y * this.zoom) * this.curve;
                this.flowField.push(angle);
            }
        }
        for (let i = 0; i < this.numberOfParticles; i++) {
            this.particles.push(new Particle(this));
        }
    }

    drawGrid(context) {
        context.save();
        context.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        context.lineWidth = 1;
        for (let c = 0; c < this.cols; c++) {
            context.beginPath();
            context.moveTo(this.cellSize * c, 0);
            context.lineTo(this.cellSize * c, this.height);
            context.stroke();
        }
        for (let r = 0; r < this.rows; r++) {
            context.beginPath();
            context.moveTo(0, this.cellSize * r);
            context.lineTo(this.width, this.cellSize * r);
            context.stroke();
        }
        context.restore();
    }

    render(context) {
        this.drawGrid(context);
        this.particles.forEach((particle) => {
            particle.draw(context);
            particle.update();
        });
    }
}

const effect = new Effect(canvas.width, canvas.height);

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    effect.render(ctx);

    ctx.font = '48px Arial';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('Welcome, MnC Guys! Go with The Flow', canvas.width / 2, canvas.height / 2);

    ctx.font = '20px Arial';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('Made by Mathematics & Computing Dept. ', canvas.width / 2, canvas.height - 20);
    
    requestAnimationFrame(animate);
}

animate();
