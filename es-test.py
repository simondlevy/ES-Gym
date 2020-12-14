#!/usr/bin/env python3
'''
ES tester script for OpenAI Gym environemnts

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

import argparse
import torch

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = torch.load(open(args.filename, 'rb'))

    print(net)
    print()
    print(env_name)

    '''
    # Run code in script named by environment
    code = open('./nets/%s.py'% args.env).read() 
    ldict = {}
    exec(code, globals(), ldict)
    net = ldict['net']

    if cuda:
        net = net.cuda()

    def copy_weights_to_net(weights, net):

        for i, param in enumerate(net.parameters()):
            try:
                param.data.copy_(weights[i])
            except:
                param.data.copy_(weights[i].data)

    def get_reward(weights, net, render=False):

        cloned_net = copy.deepcopy(net)

        copy_weights_to_net(weights, cloned_net)

        env = gym.make(args.env)
        ob = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            batch = torch.from_numpy(ob[np.newaxis,...]).float()
            if cuda:
                batch = batch.cuda()
            prediction = cloned_net(Variable(batch))
            action = prediction.data.numpy().argmax()
            ob, reward, done, _ = env.step(action)

            total_reward += reward 

        env.close()
        return total_reward
        
    partial_func = partial(get_reward, net=net)
    mother_parameters = list(net.parameters())

    es = EvolutionModule(
        mother_parameters, partial_func, population_size=args.pop, sigma=args.sigma, 
        learning_rate=args.lr, threadcount=args.threads, cuda=cuda, reward_goal=args.target,
        consecutive_goal_stopping=args.csg
    )
    final_weights = es.run(args.iter)

    # Save final weights in a new network, along with environment name
    os.makedirs('solutions', exist_ok=True)
    reward = partial_func(final_weights)
    copy_weights_to_net(final_weights, net)
    filename = 'solutions/%s%+.3f.dat' % (args.env, reward)
    print('Saving %s' % filename)
    torch.save((net,args.env), open(filename, 'wb'))

    #reward = partial_func(final_weights, render=True)
    #print(f'Reward from final weights: {reward}')
    '''

if __name__ == '__main__':
    main()
