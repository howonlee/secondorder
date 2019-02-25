Fast Newton's Method for Neural Nets, with Finite Difference
=====

With some particularly strange numerics and a very idiosyncratic requirement on the network.

This is continuing on the work I did in the CSPAM thread (memorialized outside the thread [first](https://github.com/howonlee/bobdobbshess), [second](https://github.com/howonlee/bobdobbsnewton), [third](https://github.com/howonlee/twostrangethings)) and you'd probably be best off asking question or making comments there, although I'll probably end up hanging in the HN thread for a while. If you need a really friendly explanation, you are best off asking me questions in CSPAM. I have a general inability to give understandable explanations except in reply to specific people and probably a solid inability to give understandable explanations at all for this stuff, but at least my code works, so if everything goes wahoonie you could read that.

It turns out that the actual thing preventing me from getting good second order method Hessian-inverse-premultiplied-gradients from nontrivial networks was a stupid finite difference problem. So this is the final result. It is as usual completely useless in the exact version presented to you here, but now I believe that other folks will really actually be interested in it. This is on the order of the inversion only, which if you fuss with the network a bit and the minibatches, you can get to on the order of the backpropagation, as opposed to the naive method also stuffed in the code.

The pretty strict and extremely strange requirement is that everything have the proper rank with respect to matrix multiplication for one directional inverse to continually exist. So you would be able to, depending on how you set your net up, to have monotonically increasing or decreasing cardinality of units as layers go on, but not both. I note that deconvolution is a lot easier, as inverse operators go, than actual matmuls, and the difficult part is getting inverses of functions.

The bad numerics is entirely contingent on the finite difference interpretation. Just as in the Pearlmutter method, you should be able to also get an analytic solution by actually taking the definition of the Gateaux derivative. But unlike my [previous repo](https://github.com/howonlee/twostrangethings), which had only a trivial neural net, this is a true second order nontrivial net. Only the barest of nontriviality, of course.

I now consider this damned thing finished as it will be before I work on other stuff. Next is the nth order Householder functions and trying this method on some extremely other things before actually beginning on the CSPAM project.
