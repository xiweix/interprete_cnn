<template>
  <v-container fluid>
    <p>
      Canvas
      <v-icon @click="updateImg">mdi-refresh-circle</v-icon>
      <v-icon @click="clearCanvas">mdi-alpha-c-circle</v-icon>
      <v-icon v-if="showIcon === 1" @click="saveImg(canvas)"
        >mdi-send-circle</v-icon
      >
    </p>
    <canvas
      id="canvas"
      width="150"
      height="150"
      @mousedown="startPainting"
      @mouseup="finishedPainting"
      @mousemove="draw"
    ></canvas>
  </v-container>
</template>

<script>
export default {
  name: "drawing",
  //data() {
  //    painting: false;
  //    ctx: null;
  //    canvas: null;
  //},
  data: () => ({
    painting: false,
    ctx: null,
    canvas: null,
    rect: null,
    showIcon: 0,
    OrgImg: null,
  }),
  mounted() {
    this.canvas = document.getElementById("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.canvas.height = 150;
    this.canvas.width = 150;
    this.rect = this.canvas.getBoundingClientRect();
    this.ctx.fillRect(0, 0, 150, 150);
    this.ctx.strokeStyle = "#ffffff";
    this.ctx.stroke();
    //this.vueCanvas = ctx;
    this.$options.sockets.onmessage = (res) => {
      res = res.data;
      if (res.indexOf("sample_canvas***") !== -1) {
        res = res.split("***");
        this.OrgImg = "data:image/png;base64," + res[1];
        this.updateImg();
      } else if (res.indexOf("epochSelected***") !== -1) {
        this.showIcon = 1;
      }
    };
  },
  methods: {
    startPainting(e) {
      this.painting = true;
      // console.log(this.painting);
      this.draw(e);
    },
    finishedPainting() {
      this.painting = false;
      // console.log(this.painting);
      this.ctx.beginPath();
    },
    draw(e) {
      if (!this.painting) return;
      this.rect = this.canvas.getBoundingClientRect();
      // console.log("Mouse at " + e.clientX + ", " + e.clientY);
      // console.log("Origin " + this.rect.left + ", " + this.rect.right);
      this.ctx.lineWidth = 10;
      this.ctx.lineCap = "round";
      var x = e.clientX - this.rect.left;
      var y = e.clientY - this.rect.top;
      this.ctx.lineTo(x, y);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.moveTo(x, y);
      //this.ctx.fillRect(e.clientX,e.clientY,30,30)
      //this.ctx.fillRect(10, 10,30,30)
      //console.log("Box")
    },
    saveImg(e) {
      // console.log(e.toDataURL());
      this.$socket.send("request_annotated_activations***" + e.toDataURL());
    },
    updateImg() {
      if (this.OrgImg !== null) {
        var img = new Image();
        img.src = this.OrgImg;
        img.onload = () => {
          // console.log("this.ctx:", this.ctx);
          this.ctx.drawImage(img, 0, 0, 150, 150);
        };
      } else {
        this.ctx.fillRect(0, 0, 150, 150);
      }
    },
    clearCanvas() {
      this.ctx.fillRect(0, 0, 150, 150);
    }
  },
};
</script>
