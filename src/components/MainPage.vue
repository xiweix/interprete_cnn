<template>
  <v-container fluid>
    <v-row no-gutters>
      <v-col cols="3">
        <v-card>
          <v-row no-gutters>
            <v-col width="33%">
              <v-list dense>
                <v-subheader>Model1</v-subheader>
                <v-list-item>
                  <v-text-field
                    v-model="nLayers1"
                    label="# of layers"
                    type="number"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                  <v-text-field
                    v-model="features1"
                    label="features"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                  <v-text-field
                    v-model="dropRate1"
                    label="drop out rate"
                    type="number"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                  <v-text-field
                    v-model="lr1"
                    label="learning rate"
                    type="number"
                  ></v-text-field>
                </v-list-item>
              </v-list>
            </v-col>
            <v-col width="33%">
              <v-list dense>
                <v-subheader>Model2</v-subheader>
                <v-list-item>
                  <v-text-field
                    v-model="nLayers2"
                    label="# of layers"
                    type="number"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                  <v-text-field
                    v-model="features2"
                    label="features"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                  <v-text-field
                    v-model="dropRate2"
                    label="drop out rate"
                    type="number"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                  <v-text-field
                    v-model="lr2"
                    label="learning rate"
                    type="number"
                  ></v-text-field>
                </v-list-item>
              </v-list>
            </v-col>
            <v-col width="33%">
              <v-list>
                <v-subheader>Other</v-subheader>
                <v-list-item>
                  <v-text-field
                    v-model="epochs"
                    label="epochs"
                    type="number"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                  <v-text-field
                    v-model="TrainBatchSize"
                    label="train batch size"
                    type="number"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                  <v-text-field
                    v-model="lrStepGamma"
                    label="learning rate step gamma"
                    type="number"
                  ></v-text-field>
                </v-list-item>
              </v-list>
              <v-list-item>
                <v-btn
                  color="primary"
                  text
                  @click="
                    start(
                      nLayers1,
                      features1,
                      dropRate1,
                      lr1,
                      nLayers2,
                      features2,
                      dropRate2,
                      lr2,
                      epochs,
                      TrainBatchSize,
                      lrStepGamma
                    )
                  "
                  >RUN</v-btn
                >
                <!-- <v-spacer></v-spacer>
                <v-btn
                  v-if="refresh === true"
                  color="primary"
                  text
                  @click="sync"
                  >SYNC</v-btn
                > -->
              </v-list-item>
            </v-col>
          </v-row>
        </v-card>
      </v-col>
      <v-col cols="9">
        <v-row no-gutters>
          <v-col cols="auto">
            <plotMnist />
          </v-col>
          <v-col cols="auto">
            <plotLossTrain />
            <plotAccTrain />
          </v-col>
          <v-col cols="auto">
            <plotLossTest />
            <plotAccTest />
          </v-col>
          <!-- <v-col cols="auto" v-if="refresh === true">
            <v-icon @click="sync">mdi-sync</v-icon>
          </v-col> -->
        </v-row>
      </v-col>
    </v-row>
    <v-row no-gutters>
      <v-col cols="auto">
        <Org />
      </v-col>
      <v-col>
        <activationOrg />
      </v-col>
    </v-row>
    <v-divider></v-divider>
    <v-row no-gutters>
      <v-col cols="auto">
        <drawing />
      </v-col>
      <v-col>
        <activationAnnotation />
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import plotLossTrain from "@/components/plotLossTrain.vue";
import plotAccTrain from "@/components/plotAccTrain.vue";
import plotLossTest from "@/components/plotLossTest.vue";
import plotAccTest from "@/components/plotAccTest.vue";
import plotMnist from "@/components/plotMnist.vue";
import Org from "@/components/Org.vue";
import drawing from "@/components/drawing.vue";
import activationOrg from "@/components/activationOrg.vue";
import activationAnnotation from "@/components/activationAnnotation.vue";

export default {
  components: {
    plotLossTrain,
    plotAccTrain,
    plotLossTest,
    plotAccTest,
    plotMnist,
    Org,
    drawing,
    activationOrg,
    activationAnnotation,
  },

  data: () => ({
    nLayers1: 1,
    features1: [32],
    dropRate1: -1,
    lr1: 0.001,
    nLayers2: 3,
    features2: [32, 64, 64],
    dropRate2: 0.5,
    lr2: 0.001,
    epochs: 30,
    TrainBatchSize: 256,
    lrStepGamma: 0.7,
    refresh: false,
    finish: 0,
  }),

  methods: {
    start: function(
      res1_1,
      res1_2,
      res1_3,
      res1_4,
      res2_1,
      res2_2,
      res2_3,
      res2_4,
      res3,
      res4,
      res5
    ) {
      console.log("WebSocket connection state: " + this.$socket.readyState);
      this.$socket.send(
        "start***" +
          res1_1 +
          "***" +
          res1_2 +
          "***" +
          res1_3 +
          "***" +
          res1_4 +
          "***" +
          res2_1 +
          "***" +
          res2_2 +
          "***" +
          res2_3 +
          "***" +
          res2_4 +
          "***" +
          res3 +
          "***" +
          res4 +
          "***" +
          res5
      );
    },
    reset: function() {
      this.nLayers1 = 1;
      this.features1 = [32];
      this.dropRate1 = -1;
      this.nLayers2 = 2;
      this.features2 = [32, 64];
      this.dropRate2 = 0.5;
      this.lr1 = 0.001;
      this.lr2 = 0.001;
    },
    sync: function(timer) {
      this.$socket.send("refresh***");
      setTimeout(() => {
        if (this.finish === 1) {
          clearInterval(timer)
        }
      }, 0)
    },
  },

  mounted() {
    this.$options.sockets.onmessage = (res) => {
      res = res.data;
      if (res.indexOf("start_training***") !== -1) {
        this.refresh = true;
        let timer = setInterval(() => {
          this.sync(timer)
        }, 5000)
      } else if (res.indexOf("train_finished***") !== -1) {
        this.finish = 1
      }
    };
  },
};
</script>
