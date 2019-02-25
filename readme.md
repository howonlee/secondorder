Second Order Neural Nets, with Finite Difference
=====

With some particularly insane numerics

This is continuing on the work I did in the CSPAM thread ([first](https://github.com/howonlee/bobdobbshess), [second](https://github.com/howonlee/bobdobbsnewton), [third](https://github.com/howonlee/twostrangethings)) and you'd probably be best off asking question or making comments there, although I'll probably end up hanging in the HN thread for a while. If you need a really friendly explanation, you are better off asking me questions in those places. I have a general inability to give understandable explanations except in reply to specific people.

It turns out that the actual thing preventing me from getting good second order method Hessian-inverse-premultiplied-gradients from nontrivial networks was a stupid finite difference problem. So this is the final result. As usual completely useless in the exact version presented to you here, but now I believe that other folks will really actually be interested in it.

The pretty strict and extremely strange requirement is that everything have the proper rank with respect to matrix multiplication. So you would be able to, depending on how you set your net up, to have monotonically increasing or decreasing cardinality of units as layers go on, but not both. This is to have the proper directional inverse for the linear operators. I note that deconvolution is a lot easier, as inverse operators go, than actual linear operators, and the difficult part is getting inverses of functions.

The bad numerics is entirely contingent on the finite difference interpretation, you should just like the Pearlmutter method be able to also get an analytic solution. But unlike my [previous repo](https://github.com/howonlee/twostrangethings), which had only a trivial neural net, this is a true second order nontrivial net. Only the barest of nontriviality, of course.

I now consider this damned thing finished as it will be before I work on other stuff. Next is the nth order Householder functions and trying this stuff on other things before the September project.
