---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

Note: Project ongoing

<div class="container">
    <figure>
        <img src="{{ site.baseurl}}/assets/example.png">
        <figcaption style="text-align: center; color: gray;">Left: During training, Alice will propose progressively more challenging tasks for Bob to complete. Right: On the target task, Bob must navigate to the specified point.</figcaption>
    </figure>
    <hr/>
    <br/>
    <p align="justify">
      We outline a method for training agents on nagivation tasks in realistic 3D environments. Specifically, we are training agents on pointgoal navigation and object navigation tasks, as described in the <a href="https://aihabitat.org/challenge/2020/" target="_blank">Habitat Challenge</a>. The agent uses asymmetric self-play [1] to develop a curriculum of progressively more challenging navigation goals. The agent is them fine-tuned to the navigation datasets provided by Habitat. As described in [1], the agent has two "minds"; one called Alice, which proposes progressively more challenging tasks for the other, called Bob. There is also a target task, which in our case is a navigation task, that Bob is fine-tuned on once the self-play training is complete.
    </p>
    <p style="text-align: center;">
        <!-- <a class="btn btn-primary btn-lg" href="https://arxiv.org/pdf/1703.05407" target="_blank">PDF</a> -->
        <!-- <a class="btn btn-success btn-lg" href="selfplay_poster.pdf" target="_blank">Poster</a> -->
        <a class="btn btn-default btn-lg" href="https://github.com/mpiseno/nav-selfplay" target="_blank">Code</a>
        <!-- <a class="btn btn-info btn-lg" href="https://youtu.be/5dNAnCYBFN4" target="_blank">Talk</a> -->
    </p>
    <hr/>
    <!-- <div class="text-center">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/EHHiFwStqaA?rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
    </div> -->
    <hr/>
    <br/>
    <h3 style="text-align: center; color: black;">References</h3>
    <p>
        [1] Sainbayar Sukhbaatar et al. 2018. Intrinsic Motivation And Automatic Curricula Via Asymmetric Self-Play. arXiv: 1703.05407. Retrieved from https://arxiv.org/pdf/1703.05407.pdf
    </p>
</div>

<!-- Bootstrap core JavaScript
================================================== -->
<!-- Placed at the end of the document so the pages load faster -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.min.js"></script>
<!-- <script src="js/bootstrap.min.js"></script> -->

<!-- google analyitcs -->
<script>
(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
    (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
    m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
})(window,document,'script','//www.google-analytics.com/analytics.js','ga');

ga('create', 'UA-37493933-2', 'auto');
ga('send', 'pageview');

</script>

